impl CellGrid {
    /// Generates the cell grid, given the tracks and cells.
    pub fn new(
        tracks: Axes<&[Sizing]>,
        gutter: Axes<&[Sizing]>,
        cells: impl IntoIterator<Item = Cell>,
    ) -> Self {
        let entries = cells.into_iter().map(Entry::Cell).collect();
        Self::new_internal(tracks, gutter, vec![], vec![], None, None, entries)
    }

    /// Resolves and positions all cells in the grid before creating it.
    /// Allows them to keep track of their final properties and positions
    /// and adjust their fields accordingly.
    /// Cells must implement Clone as they will be owned. Additionally, they
    /// must implement Default in order to fill positions in the grid which
    /// weren't explicitly specified by the user with empty cells.
    #[allow(clippy::too_many_arguments)]
    pub fn resolve<T, C, I>(
        tracks: Axes<&[Sizing]>,
        gutter: Axes<&[Sizing]>,
        children: C,
        fill: &Celled<Option<Paint>>,
        align: &Celled<Smart<Alignment>>,
        inset: &Celled<Sides<Option<Rel<Length>>>>,
        stroke: &ResolvedCelled<Sides<Option<Option<Arc<Stroke>>>>>,
        engine: &mut Engine,
        styles: StyleChain,
        span: Span,
    ) -> SourceResult<Self>
    where
        T: ResolvableCell + Default,
        I: Iterator<Item = ResolvableGridItem<T>>,
        C: IntoIterator<Item = ResolvableGridChild<T, I>>,
        C::IntoIter: ExactSizeIterator,
    {
        // Number of content columns: Always at least one.
        let c = tracks.x.len().max(1);

        // Lists of lines.
        // Horizontal lines are only pushed later to be able to check for row
        // validity, since the amount of rows isn't known until all items were
        // analyzed in the for loop below.
        // We keep their spans so we can report errors later.
        // The additional boolean indicates whether the hline had an automatic
        // 'y' index, and is used to change the index of hlines at the top of a
        // header or footer.
        let mut pending_hlines: Vec<(Span, Line, bool)> = vec![];

        // For consistency, only push vertical lines later as well.
        let mut pending_vlines: Vec<(Span, Line)> = vec![];
        let has_gutter = gutter.any(|tracks| !tracks.is_empty());

        let mut header: Option<Header> = None;
        let mut repeat_header = false;

        // Stores where the footer is supposed to end, its span, and the
        // actual footer structure.
        let mut footer: Option<(usize, Span, Footer)> = None;
        let mut repeat_footer = false;

        // Resolve the breakability of a cell, based on whether or not it spans
        // an auto row.
        let resolve_breakable = |y, rowspan| {
            let auto = Sizing::Auto;
            let zero = Sizing::Rel(Rel::zero());
            tracks
                .y
                .iter()
                .chain(std::iter::repeat(tracks.y.last().unwrap_or(&auto)))
                .skip(y)
                .take(rowspan)
                .any(|row| row == &Sizing::Auto)
                || gutter
                    .y
                    .iter()
                    .chain(std::iter::repeat(gutter.y.last().unwrap_or(&zero)))
                    .skip(y)
                    .take(rowspan - 1)
                    .any(|row_gutter| row_gutter == &Sizing::Auto)
        };

        // We can't just use the cell's index in the 'cells' vector to
        // determine its automatic position, since cells could have arbitrary
        // positions, so the position of a cell in 'cells' can differ from its
        // final position in 'resolved_cells' (see below).
        // Therefore, we use a counter, 'auto_index', to determine the position
        // of the next cell with (x: auto, y: auto). It is only stepped when
        // a cell with (x: auto, y: auto), usually the vast majority, is found.
        let mut auto_index: usize = 0;

        // We have to rebuild the grid to account for arbitrary positions.
        // Create at least 'children.len()' positions, since there could be at
        // least 'children.len()' cells (if no explicit lines were specified),
        // even though some of them might be placed in arbitrary positions and
        // thus cause the grid to expand.
        // Additionally, make sure we allocate up to the next multiple of 'c',
        // since each row will have 'c' cells, even if the last few cells
        // weren't explicitly specified by the user.
        // We apply '% c' twice so that the amount of cells potentially missing
        // is zero when 'children.len()' is already a multiple of 'c' (thus
        // 'children.len() % c' would be zero).
        let children = children.into_iter();
        let Some(child_count) = children.len().checked_add((c - children.len() % c) % c)
        else {
            bail!(span, "too many cells or lines were given")
        };
        let mut resolved_cells: Vec<Option<Entry>> = Vec::with_capacity(child_count);
        for child in children {
            let mut is_header = false;
            let mut is_footer = false;
            let mut child_start = usize::MAX;
            let mut child_end = 0;
            let mut child_span = Span::detached();
            let mut start_new_row = false;
            let mut first_index_of_top_hlines = usize::MAX;
            let mut first_index_of_non_top_hlines = usize::MAX;

            let (header_footer_items, simple_item) = match child {
                ResolvableGridChild::Header { repeat, span, items, .. } => {
                    if header.is_some() {
                        bail!(span, "cannot have more than one header");
                    }

                    is_header = true;
                    child_span = span;
                    repeat_header = repeat;

                    // If any cell in the header is automatically positioned,
                    // have it skip to the next row. This is to avoid having a
                    // header after a partially filled row just add cells to
                    // that row instead of starting a new one.
                    // FIXME: Revise this approach when headers can start from
                    // arbitrary rows.
                    start_new_row = true;

                    // Any hlines at the top of the header will start at this
                    // index.
                    first_index_of_top_hlines = pending_hlines.len();

                    (Some(items), None)
                }
                ResolvableGridChild::Footer { repeat, span, items, .. } => {
                    if footer.is_some() {
                        bail!(span, "cannot have more than one footer");
                    }

                    is_footer = true;
                    child_span = span;
                    repeat_footer = repeat;

                    // If any cell in the footer is automatically positioned,
                    // have it skip to the next row. This is to avoid having a
                    // footer after a partially filled row just add cells to
                    // that row instead of starting a new one.
                    start_new_row = true;

                    // Any hlines at the top of the footer will start at this
                    // index.
                    first_index_of_top_hlines = pending_hlines.len();

                    (Some(items), None)
                }
                ResolvableGridChild::Item(item) => (None, Some(item)),
            };

            let items = header_footer_items
                .into_iter()
                .flatten()
                .chain(simple_item.into_iter());
            for item in items {
                let cell = match item {
                    ResolvableGridItem::HLine {
                        y,
                        start,
                        end,
                        stroke,
                        span,
                        position,
                    } => {
                        let has_auto_y = y.is_auto();
                        let y = y.unwrap_or_else(|| {
                            // Avoid placing the hline inside consecutive
                            // rowspans occupying all columns, as it'd just
                            // disappear, at least when there's no column
                            // gutter.
                            skip_auto_index_through_fully_merged_rows(
                                &resolved_cells,
                                &mut auto_index,
                                c,
                            );

                            // When no 'y' is specified for the hline, we place
                            // it under the latest automatically positioned
                            // cell.
                            // The current value of the auto index is always
                            // the index of the latest automatically positioned
                            // cell placed plus one (that's what we do in
                            // 'resolve_cell_position'), so we subtract 1 to
                            // get that cell's index, and place the hline below
                            // its row. The exception is when the auto_index is
                            // 0, meaning no automatically positioned cell was
                            // placed yet. In that case, we place the hline at
                            // the top of the table.
                            //
                            // Exceptionally, the hline will be placed before
                            // the minimum auto index if the current auto index
                            // from previous iterations is smaller than the
                            // minimum it should have for the current grid
                            // child. Effectively, this means that a hline at
                            // the start of a header will always appear above
                            // that header's first row. Similarly for footers.
                            auto_index
                                .checked_sub(1)
                                .map_or(0, |last_auto_index| last_auto_index / c + 1)
                        });
                        if end.is_some_and(|end| end.get() < start) {
                            bail!(span, "line cannot end before it starts");
                        }
                        let line = Line { index: y, start, end, stroke, position };

                        // Since the amount of rows is dynamic, delay placing
                        // hlines until after all cells were placed so we can
                        // properly verify if they are valid. Note that we
                        // can't place hlines even if we already know they
                        // would be in a valid row, since it's possible that we
                        // pushed pending hlines in the same row as this one in
                        // previous iterations, and we need to ensure that
                        // hlines from previous iterations are pushed to the
                        // final vector of hlines first - the order of hlines
                        // must be kept, as this matters when determining which
                        // one "wins" in case of conflict. Pushing the current
                        // hline before we push pending hlines later would
                        // change their order!
                        pending_hlines.push((span, line, has_auto_y));
                        continue;
                    }
                    ResolvableGridItem::VLine {
                        x,
                        start,
                        end,
                        stroke,
                        span,
                        position,
                    } => {
                        let x = x.unwrap_or_else(|| {
                            // When no 'x' is specified for the vline, we place
                            // it after the latest automatically positioned
                            // cell.
                            // The current value of the auto index is always
                            // the index of the latest automatically positioned
                            // cell placed plus one (that's what we do in
                            // 'resolve_cell_position'), so we subtract 1 to
                            // get that cell's index, and place the vline after
                            // its column. The exception is when the auto_index
                            // is 0, meaning no automatically positioned cell
                            // was placed yet. In that case, we place the vline
                            // to the left of the table.
                            //
                            // Exceptionally, a vline is also placed to the
                            // left of the table if we should start a new row
                            // for the next automatically positioned cell.
                            // For example, this means that a vline at
                            // the beginning of a header will be placed to its
                            // left rather than after the previous
                            // automatically positioned cell. Same for footers.
                            auto_index
                                .checked_sub(1)
                                .filter(|_| !start_new_row)
                                .map_or(0, |last_auto_index| last_auto_index % c + 1)
                        });
                        if end.is_some_and(|end| end.get() < start) {
                            bail!(span, "line cannot end before it starts");
                        }
                        let line = Line { index: x, start, end, stroke, position };

                        // For consistency with hlines, we only push vlines to
                        // the final vector of vlines after processing every
                        // cell.
                        pending_vlines.push((span, line));
                        continue;
                    }
                    ResolvableGridItem::Cell(cell) => cell,
                };
                let cell_span = cell.span();
                let colspan = cell.colspan(styles).get();
                let rowspan = cell.rowspan(styles).get();
                // Let's calculate the cell's final position based on its
                // requested position.
                let resolved_index = {
                    let cell_x = cell.x(styles);
                    let cell_y = cell.y(styles);
                    resolve_cell_position(
                        cell_x,
                        cell_y,
                        colspan,
                        rowspan,
                        &resolved_cells,
                        &mut auto_index,
                        &mut start_new_row,
                        c,
                    )
                    .at(cell_span)?
                };
                let x = resolved_index % c;
                let y = resolved_index / c;

                if colspan > c - x {
                    bail!(
                        cell_span,
                        "cell's colspan would cause it to exceed the available column(s)";
                        hint: "try placing the cell in another position or reducing its colspan"
                    )
                }

                let Some(largest_index) = c
                    .checked_mul(rowspan - 1)
                    .and_then(|full_rowspan_offset| {
                        resolved_index.checked_add(full_rowspan_offset)
                    })
                    .and_then(|last_row_pos| last_row_pos.checked_add(colspan - 1))
                else {
                    bail!(
                        cell_span,
                        "cell would span an exceedingly large position";
                        hint: "try reducing the cell's rowspan or colspan"
                    )
                };

                // Let's resolve the cell so it can determine its own fields
                // based on its final position.
                let cell = cell.resolve_cell(
                    x,
                    y,
                    &fill.resolve(engine, styles, x, y)?,
                    align.resolve(engine, styles, x, y)?,
                    inset.resolve(engine, styles, x, y)?,
                    stroke.resolve(engine, styles, x, y)?,
                    resolve_breakable(y, rowspan),
                    styles,
                );

                if largest_index >= resolved_cells.len() {
                    // Ensure the length of the vector of resolved cells is
                    // always a multiple of 'c' by pushing full rows every
                    // time. Here, we add enough absent positions (later
                    // converted to empty cells) to ensure the last row in the
                    // new vector length is completely filled. This is
                    // necessary so that those positions, even if not
                    // explicitly used at the end, are eventually susceptible
                    // to show rules and receive grid styling, as they will be
                    // resolved as empty cells in a second loop below.
                    let Some(new_len) = largest_index
                        .checked_add(1)
                        .and_then(|new_len| new_len.checked_add((c - new_len % c) % c))
                    else {
                        bail!(cell_span, "cell position too large")
                    };

                    // Here, the cell needs to be placed in a position which
                    // doesn't exist yet in the grid (out of bounds). We will
                    // add enough absent positions for this to be possible.
                    // They must be absent as no cells actually occupy them
                    // (they can be overridden later); however, if no cells
                    // occupy them as we finish building the grid, then such
                    // positions will be replaced by empty cells.
                    resolved_cells.resize(new_len, None);
                }

                // The vector is large enough to contain the cell, so we can
                // just index it directly to access the position it will be
                // placed in. However, we still need to ensure we won't try to
                // place a cell where there already is one.
                let slot = &mut resolved_cells[resolved_index];
                if slot.is_some() {
                    bail!(
                        cell_span,
                        "attempted to place a second cell at column {x}, row {y}";
                        hint: "try specifying your cells in a different order"
                    );
                }

                *slot = Some(Entry::Cell(cell));

                // Now, if the cell spans more than one row or column, we fill
                // the spanned positions in the grid with Entry::Merged
                // pointing to the original cell as its parent.
                for rowspan_offset in 0..rowspan {
                    let spanned_y = y + rowspan_offset;
                    let first_row_index = resolved_index + c * rowspan_offset;
                    for (colspan_offset, slot) in resolved_cells[first_row_index..]
                        [..colspan]
                        .iter_mut()
                        .enumerate()
                    {
                        let spanned_x = x + colspan_offset;
                        if spanned_x == x && spanned_y == y {
                            // This is the parent cell.
                            continue;
                        }
                        if slot.is_some() {
                            bail!(
                                cell_span,
                                "cell would span a previously placed cell at column {spanned_x}, row {spanned_y}";
                                hint: "try specifying your cells in a different order or reducing the cell's rowspan or colspan"
                            )
                        }
                        *slot = Some(Entry::Merged { parent: resolved_index });
                    }
                }

                if is_header || is_footer {
                    // Ensure each cell in a header or footer is fully
                    // contained within it.
                    child_start = child_start.min(y);
                    child_end = child_end.max(y + rowspan);

                    if start_new_row && child_start <= auto_index.div_ceil(c) {
                        // No need to start a new row as we already include
                        // the row of the next automatically positioned cell in
                        // the header or footer.
                        start_new_row = false;
                    }

                    if !start_new_row {
                        // From now on, upcoming hlines won't be at the top of
                        // the child, as the first automatically positioned
                        // cell was placed.
                        first_index_of_non_top_hlines =
                            first_index_of_non_top_hlines.min(pending_hlines.len());
                    }
                }
            }

            if (is_header || is_footer) && child_start == usize::MAX {
                // Empty header/footer: consider the header/footer to be
                // at the next empty row after the latest auto index.
                auto_index = find_next_empty_row(&resolved_cells, auto_index, c);
                child_start = auto_index.div_ceil(c);
                child_end = child_start + 1;

                if resolved_cells.len() <= c * child_start {
                    // Ensure the automatically chosen row actually exists.
                    resolved_cells.resize_with(c * (child_start + 1), || None);
                }
            }

            if is_header {
                if child_start != 0 {
                    bail!(
                        child_span,
                        "header must start at the first row";
                        hint: "remove any rows before the header"
                    );
                }

                header = Some(Header {
                    // Later on, we have to correct this number in case there
                    // is gutter. But only once all cells have been analyzed
                    // and the header has fully expanded in the fixup loop
                    // below.
                    end: child_end,
                });
            }

            if is_footer {
                // Only check if the footer is at the end later, once we know
                // the final amount of rows.
                footer = Some((
                    child_end,
                    child_span,
                    Footer {
                        // Later on, we have to correct this number in case there
                        // is gutter, but only once all cells have been analyzed
                        // and the header's and footer's exact boundaries are
                        // known. That is because the gutter row immediately
                        // before the footer might not be included as part of
                        // the footer if it is contained within the header.
                        start: child_start,
                    },
                ));
            }

            if is_header || is_footer {
                let amount_hlines = pending_hlines.len();
                for (_, top_hline, has_auto_y) in pending_hlines
                    .get_mut(
                        first_index_of_top_hlines
                            ..first_index_of_non_top_hlines.min(amount_hlines),
                    )
                    .unwrap_or(&mut [])
                {
                    if *has_auto_y {
                        // Move this hline to the top of the child, as it was
                        // placed before the first automatically positioned cell
                        // and had an automatic index.
                        top_hline.index = child_start;
                    }
                }

                // Next automatically positioned cell goes under this header.
                // FIXME: Consider only doing this if the header has any fully
                // automatically positioned cells. Otherwise,
                // `resolve_cell_position` should be smart enough to skip
                // upcoming headers.
                // Additionally, consider that cells with just an 'x' override
                // could end up going too far back and making previous
                // non-header rows into header rows (maybe they should be
                // placed at the first row that is fully empty or something).
                // Nothing we can do when both 'x' and 'y' were overridden, of
                // course.
                // None of the above are concerns for now, as headers must
                // start at the first row.
                auto_index = auto_index.max(c * child_end);
            }
        }

        // If the user specified cells occupying less rows than the given rows,
        // we shall expand the grid so that it has at least the given amount of
        // rows.
        let Some(expected_total_cells) = c.checked_mul(tracks.y.len()) else {
            bail!(span, "too many rows were specified");
        };
        let missing_cells = expected_total_cells.saturating_sub(resolved_cells.len());

        // Fixup phase (final step in cell grid generation):
        // 1. Replace absent entries by resolved empty cells, and produce a
        // vector of 'Entry' from 'Option<Entry>'.
        // 2. Add enough empty cells to the end of the grid such that it has at
        // least the given amount of rows.
        // 3. If any cells were added to the header's rows after the header's
        // creation, ensure the header expands enough to accommodate them
        // across all of their spanned rows. Same for the footer.
        // 4. If any cells before the footer try to span it, error.
        let resolved_cells = resolved_cells
            .into_iter()
            .chain(std::iter::repeat_with(|| None).take(missing_cells))
            .enumerate()
            .map(|(i, cell)| {
                if let Some(cell) = cell {
                    if let Some(parent_cell) = cell.as_cell() {
                        if let Some(header) = &mut header
                        {
                            let y = i / c;
                            if y < header.end {
                                // Ensure the header expands enough such that
                                // all cells inside it, even those added later,
                                // are fully contained within the header.
                                // FIXME: check if start < y < end when start can
                                // be != 0.
                                // FIXME: when start can be != 0, decide what
                                // happens when a cell after the header placed
                                // above it tries to span the header (either
                                // error or expand upwards).
                                header.end = header.end.max(y + parent_cell.rowspan.get());
                            }
                        }

                        if let Some((end, footer_span, footer)) = &mut footer {
                            let x = i % c;
                            let y = i / c;
                            let cell_end = y + parent_cell.rowspan.get();
                            if y < footer.start && cell_end > footer.start {
                                // Don't allow a cell before the footer to span
                                // it. Surely, we could move the footer to
                                // start at where this cell starts, so this is
                                // more of a design choice, as it's unlikely
                                // for the user to intentionally include a cell
                                // before the footer spanning it but not
                                // being repeated with it.
                                bail!(
                                    *footer_span,
                                    "footer would conflict with a cell placed before it at column {x} row {y}";
                                    hint: "try reducing that cell's rowspan or moving the footer"
                                );
                            }
                            if y >= footer.start && y < *end {
                                // Expand the footer to include all rows
                                // spanned by this cell, as it is inside the
                                // footer.
                                *end = (*end).max(cell_end);
                            }
                        }
                    }

                    Ok(cell)
                } else {
                    let x = i % c;
                    let y = i / c;

                    // Ensure all absent entries are affected by show rules and
                    // grid styling by turning them into resolved empty cells.
                    let new_cell = T::default().resolve_cell(
                        x,
                        y,
                        &fill.resolve(engine, styles, x, y)?,
                        align.resolve(engine, styles, x, y)?,
                        inset.resolve(engine, styles, x, y)?,
                        stroke.resolve(engine, styles, x, y)?,
                        resolve_breakable(y, 1),
                        styles,
                    );
                    Ok(Entry::Cell(new_cell))
                }
            })
            .collect::<SourceResult<Vec<Entry>>>()?;

        // Populate the final lists of lines.
        // For each line type (horizontal or vertical), we keep a vector for
        // every group of lines with the same index.
        let mut vlines: Vec<Vec<Line>> = vec![];
        let mut hlines: Vec<Vec<Line>> = vec![];
        let row_amount = resolved_cells.len().div_ceil(c);

        for (line_span, line, _) in pending_hlines {
            let y = line.index;
            if y > row_amount {
                bail!(line_span, "cannot place horizontal line at invalid row {y}");
            }
            if y == row_amount && line.position == LinePosition::After {
                bail!(
                    line_span,
                    "cannot place horizontal line at the 'bottom' position of the bottom border (y = {y})";
                    hint: "set the line's position to 'top' or place it at a smaller 'y' index"
                );
            }
            let line = if line.position == LinePosition::After
                && (!has_gutter || y + 1 == row_amount)
            {
                // Just place the line on top of the next row if
                // there's no gutter and the line should be placed
                // after the one with given index.
                //
                // Note that placing after the last row is also the same as
                // just placing on the grid's bottom border, even with
                // gutter.
                Line {
                    index: y + 1,
                    position: LinePosition::Before,
                    ..line
                }
            } else {
                line
            };
            let y = line.index;

            if hlines.len() <= y {
                hlines.resize_with(y + 1, Vec::new);
            }
            hlines[y].push(line);
        }

        for (line_span, line) in pending_vlines {
            let x = line.index;
            if x > c {
                bail!(line_span, "cannot place vertical line at invalid column {x}");
            }
            if x == c && line.position == LinePosition::After {
                bail!(
                    line_span,
                    "cannot place vertical line at the 'end' position of the end border (x = {c})";
                    hint: "set the line's position to 'start' or place it at a smaller 'x' index"
                );
            }
            let line =
                if line.position == LinePosition::After && (!has_gutter || x + 1 == c) {
                    // Just place the line before the next column if
                    // there's no gutter and the line should be placed
                    // after the one with given index.
                    //
                    // Note that placing after the last column is also the
                    // same as just placing on the grid's end border, even
                    // with gutter.
                    Line {
                        index: x + 1,
                        position: LinePosition::Before,
                        ..line
                    }
                } else {
                    line
                };
            let x = line.index;

            if vlines.len() <= x {
                vlines.resize_with(x + 1, Vec::new);
            }
            vlines[x].push(line);
        }

        let header = header
            .map(|mut header| {
                // Repeat the gutter below a header (hence why we don't
                // subtract 1 from the gutter case).
                // Don't do this if there are no rows under the header.
                if has_gutter {
                    // - 'header.end' is always 'last y + 1'. The header stops
                    // before that row.
                    // - Therefore, '2 * header.end' will be 2 * (last y + 1),
                    // which is the adjusted index of the row before which the
                    // header stops, meaning it will still stop right before it
                    // even with gutter thanks to the multiplication below.
                    // - This means that it will span all rows up to
                    // '2 * (last y + 1) - 1 = 2 * last y + 1', which equates
                    // to the index of the gutter row right below the header,
                    // which is what we want (that gutter spacing should be
                    // repeated across pages to maintain uniformity).
                    header.end *= 2;

                    // If the header occupies the entire grid, ensure we don't
                    // include an extra gutter row when it doesn't exist, since
                    // the last row of the header is at the very bottom,
                    // therefore '2 * last y + 1' is not a valid index.
                    let row_amount = (2 * row_amount).saturating_sub(1);
                    header.end = header.end.min(row_amount);
                }
                header
            })
            .map(|header| {
                if repeat_header {
                    Repeatable::Repeated(header)
                } else {
                    Repeatable::NotRepeated(header)
                }
            });

        let footer = footer
            .map(|(footer_end, footer_span, mut footer)| {
                if footer_end != row_amount {
                    bail!(footer_span, "footer must end at the last row");
                }

                let header_end =
                    header.as_ref().map(Repeatable::unwrap).map(|header| header.end);

                if has_gutter {
                    // Convert the footer's start index to post-gutter coordinates.
                    footer.start *= 2;

                    // Include the gutter right before the footer, unless there is
                    // none, or the gutter is already included in the header (no
                    // rows between the header and the footer).
                    if header_end.map_or(true, |header_end| header_end != footer.start) {
                        footer.start = footer.start.saturating_sub(1);
                    }
                }

                if header_end.is_some_and(|header_end| header_end > footer.start) {
                    bail!(footer_span, "header and footer must not have common rows");
                }

                Ok(footer)
            })
            .transpose()?
            .map(|footer| {
                if repeat_footer {
                    Repeatable::Repeated(footer)
                } else {
                    Repeatable::NotRepeated(footer)
                }
            });

        Ok(Self::new_internal(
            tracks,
            gutter,
            vlines,
            hlines,
            header,
            footer,
            resolved_cells,
        ))
    }

    /// Generates the cell grid, given the tracks and resolved entries.
    pub(super) fn new_internal(
        tracks: Axes<&[Sizing]>,
        gutter: Axes<&[Sizing]>,
        vlines: Vec<Vec<Line>>,
        hlines: Vec<Vec<Line>>,
        header: Option<Repeatable<Header>>,
        footer: Option<Repeatable<Footer>>,
        entries: Vec<Entry>,
    ) -> Self {
        let mut cols = vec![];
        let mut rows = vec![];

        // Number of content columns: Always at least one.
        let c = tracks.x.len().max(1);

        // Number of content rows: At least as many as given, but also at least
        // as many as needed to place each item.
        let r = {
            let len = entries.len();
            let given = tracks.y.len();
            let needed = len / c + (len % c).clamp(0, 1);
            given.max(needed)
        };

        let has_gutter = gutter.any(|tracks| !tracks.is_empty());
        let auto = Sizing::Auto;
        let zero = Sizing::Rel(Rel::zero());
        let get_or = |tracks: &[_], idx, default| {
            tracks.get(idx).or(tracks.last()).copied().unwrap_or(default)
        };

        // Collect content and gutter columns.
        for x in 0..c {
            cols.push(get_or(tracks.x, x, auto));
            if has_gutter {
                cols.push(get_or(gutter.x, x, zero));
            }
        }

        // Collect content and gutter rows.
        for y in 0..r {
            rows.push(get_or(tracks.y, y, auto));
            if has_gutter {
                rows.push(get_or(gutter.y, y, zero));
            }
        }

        // Remove superfluous gutter tracks.
        if has_gutter {
            cols.pop();
            rows.pop();
        }

        Self {
            cols,
            rows,
            entries,
            vlines,
            hlines,
            header,
            footer,
            has_gutter,
        }
    }