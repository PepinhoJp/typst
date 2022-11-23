use std::path::{Path, PathBuf};

use comemo::{Prehashed, Track, Tracked};
use iai::{black_box, main, Iai};
use typst::diag::{FileError, FileResult};
use typst::font::{Font, FontBook};
use typst::syntax::{Source, SourceId, TokenMode, Tokens};
use typst::util::Buffer;
use typst::{Config, World};
use unscanny::Scanner;

const TEXT: &str = include_str!("../typ/benches/bench.typ");
const FONT: &[u8] = include_bytes!("../fonts/IBMPlexSans-Regular.ttf");

main!(
    bench_decode,
    bench_scan,
    bench_tokenize,
    bench_parse,
    bench_edit,
    bench_eval,
    bench_typeset,
    bench_highlight,
    bench_render,
);

fn bench_decode(iai: &mut Iai) {
    iai.run(|| {
        // We don't use chars().count() because that has a special
        // superfast implementation.
        let mut count = 0;
        let mut chars = black_box(TEXT).chars();
        while let Some(_) = chars.next() {
            count += 1;
        }
        count
    })
}

fn bench_scan(iai: &mut Iai) {
    iai.run(|| {
        let mut count = 0;
        let mut scanner = Scanner::new(black_box(TEXT));
        while let Some(_) = scanner.eat() {
            count += 1;
        }
        count
    })
}

fn bench_tokenize(iai: &mut Iai) {
    iai.run(|| Tokens::new(black_box(TEXT), black_box(TokenMode::Markup)).count());
}

fn bench_parse(iai: &mut Iai) {
    iai.run(|| typst::syntax::parse(TEXT));
}

fn bench_edit(iai: &mut Iai) {
    let mut source = Source::detached(TEXT);
    iai.run(|| black_box(source.edit(1168..1171, "_Uhr_")));
}

fn bench_highlight(iai: &mut Iai) {
    let source = Source::detached(TEXT);
    iai.run(|| {
        typst::syntax::highlight::highlight_categories(
            source.root(),
            0..source.len_bytes(),
            &mut |_, _| {},
        )
    });
}

fn bench_eval(iai: &mut Iai) {
    let world = BenchWorld::new();
    let route = typst::model::Route::default();
    iai.run(|| typst::model::eval(world.track(), route.track(), &world.source).unwrap());
}

fn bench_typeset(iai: &mut Iai) {
    let world = BenchWorld::new();
    iai.run(|| typst::typeset(&world, &world.source));
}

fn bench_render(iai: &mut Iai) {
    let world = BenchWorld::new();
    let frames = typst::typeset(&world, &world.source).unwrap();
    iai.run(|| typst::export::render(&frames[0], 1.0))
}

struct BenchWorld {
    config: Prehashed<Config>,
    book: Prehashed<FontBook>,
    font: Font,
    source: Source,
}

impl BenchWorld {
    fn new() -> Self {
        let config = Config {
            root: PathBuf::new(),
            scope: typst_library::scope(),
            styles: typst_library::styles(),
            items: typst_library::items(),
        };

        let font = Font::new(FONT.into(), 0).unwrap();
        let book = FontBook::from_fonts([&font]);
        let source = Source::detached(TEXT);

        Self {
            config: Prehashed::new(config),
            book: Prehashed::new(book),
            font,
            source,
        }
    }

    fn track(&self) -> Tracked<dyn World> {
        (self as &dyn World).track()
    }
}

impl World for BenchWorld {
    fn config(&self) -> &Prehashed<Config> {
        &self.config
    }

    fn book(&self) -> &Prehashed<FontBook> {
        &self.book
    }

    fn font(&self, _: usize) -> Option<Font> {
        Some(self.font.clone())
    }

    fn file(&self, path: &Path) -> FileResult<Buffer> {
        Err(FileError::NotFound(path.into()))
    }

    fn resolve(&self, path: &Path) -> FileResult<SourceId> {
        Err(FileError::NotFound(path.into()))
    }

    fn source(&self, _: SourceId) -> &Source {
        unimplemented!()
    }
}