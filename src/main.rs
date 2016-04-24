#![feature(zero_one)]

extern crate image;
extern crate num_traits;

use std::path::Path;

use image::GenericImage;
use image::Pixel;
use image::ImageBuffer;
use std::ops::Index;
use std::ops::AddAssign;
use num_traits::cast::ToPrimitive;
use std::num::One;
use std::env;

// ==================== Pixels ====================

/// Represents generic sequence of pixels
trait PixelLine {
	/// Type of Pixel - as in GenericImage
	type Pixel: Pixel;

	/// Returns length of the sequence
	fn len(&self) -> u32;

	/// Returns copy of the pixel at given position
	fn get_pixel(&self, index: u32) -> Self::Pixel;
}

/// Represents one column of pixels in the image
struct PixelColumn<'a, I: 'a> where I: GenericImage {
	column_index: u32,
	img: &'a I,
}

/// Represents one row of pixels in the image
struct PixelRow<'a, I: 'a> where I: GenericImage {
	row_index: u32,
	img: &'a I,
}

/// Iterator over pixels in `PixelLine`
struct PixelLineIterator<'a, P: 'a> where P: Pixel {
	index: u32,
	line: &'a PixelLine<Pixel=P>,
}

impl<'a, P> PixelLineIterator<'a, P> where P: Pixel {
	/// Constructs new PixelLineIterator from PixelLine starting at 0
	fn new(pixel_line: &'a PixelLine<Pixel=P>) -> Self {
		PixelLineIterator {index: 0, line: pixel_line}
	}
}

impl<'a, I> PixelColumn<'a, I> where I: GenericImage {
	/// Constructs new PixelColumn from a GenericImage and a column number
	/// 
	/// # Panics
	/// Panics if column is outside image boundaries.
	///
	fn new(img: &'a I, column: u32) -> PixelColumn<'a, I> {
		// column must be in range
		assert!(column < img.width());

		PixelColumn {column_index: column, img: img}
	}
}

impl<'a, I> PixelRow<'a, I> where I: GenericImage {
	/// Constructs new PixelRow from a GenericImage and a row number
	/// 
	/// # Panics
	/// Panics if row is outside image boundaries.
	///
	fn new(img :&'a I, row: u32) -> PixelRow<'a, I> {
		assert!(row < img.height());

		PixelRow {row_index: row, img: img}
	}
}

impl<'a, I> PixelLine for PixelColumn<'a, I> where I: GenericImage {
	type Pixel = I::Pixel;

	fn len(&self) -> u32 {
		self.img.height()
	}

	fn get_pixel(&self, index: u32) -> Self::Pixel {
		self.img.get_pixel(self.column_index, index)
	}
}

impl<'a, I> PixelLine for PixelRow<'a, I> where I: GenericImage {
	type Pixel = I::Pixel;

	fn len(&self) -> u32 {
		self.img.width()
	}

	fn get_pixel(&self, index: u32) -> Self::Pixel {
		self.img.get_pixel(index, self.row_index)
	}
}

impl<'a, P> Iterator for PixelLineIterator<'a, P> where P: Pixel {
	type Item = P;

	fn next(&mut self) -> Option<Self::Item> {

		if self.index < self.line.len() {
			let old = self.index;
			self.index += 1;
			Some(self.line.get_pixel(old))
		} else {
			None
		}
	}
}

/// Calculates matching score of two `PixelLines`.
///
/// The matching score is just average difference of two adjacent pixels,
/// so the less score is, the more similar lines are. Zero means the lines are identical.
///
/// # Panics
/// Panics if lines are of different length.
///
fn line_match_score<'a, P>(line1: &'a PixelLine<Pixel=P>, line2: &'a PixelLine<Pixel=P>) -> f64 where P: Pixel {
	assert!(line1.len() == line2.len());
	assert!(line1.len() > 0);

	let mut difference: u64 = 0;
	{
		let li1: PixelLineIterator<'a, P> = PixelLineIterator::new(line1);
		let li2: PixelLineIterator<'a, P> = PixelLineIterator::new(line2);

		for pixels in li1.zip(li2) {
			let (p1, p2) = pixels;

			for values in p1.channels().iter().zip(p2.channels().iter()) {
				let (v1, v2) = values;
				difference += if v1 > v2 {
					*v1 - *v2
				} else {
					*v2 - *v1
				}.to_u64().unwrap();
			}
		}
	}

	(difference as f64) / (line1.len() as f64) / (P::channel_count() as f64)
}

// ==================== Arg parsing ====================

/// Like `std::ops::Range`, but end is inclusive
struct InclusiveRange<T>  where T: AddAssign + PartialOrd + One + Copy {
	begin: T,
	end: T,
}

/// Represents iterator over values within `InclusiveRange`
struct InclusiveRangeIterator<T> where T: AddAssign + PartialOrd + One + Copy {
	index: T,
	end: T,
}

impl<T> InclusiveRange<T> where T: AddAssign + PartialOrd + One + Copy {
	/// Returns iterator starting at begin
	fn iter(&self) -> InclusiveRangeIterator<T> {
		InclusiveRangeIterator::<T> {index: self.begin, end: self.end}
	}
}

impl<T> Iterator for InclusiveRangeIterator<T> where T: AddAssign + PartialOrd + One + Copy {
	type Item = T;

	fn next(&mut self) -> Option<Self::Item> {
		if self.index <= self.end {
			let old = self.index;
			self.index += T::one();

			Some(old)
		} else {
			None
		}
	}
}

/// Convenience type cast
type InclusiveRangeU32 = InclusiveRange<u32>;

/// Represents component of input string
enum InputPart {
	String(String),
	Range(InclusiveRangeU32),
}

impl InputPart {
	/// Basically unwrap() for String
	fn take_mandatory_string(self) -> String {
		if let InputPart::String(s) = self { s } else { panic!("String is mandatory at this place") }
	}

	/// Basically unwrap() for Range
	fn take_mandatory_range(self) -> InclusiveRangeU32 {
		if let InputPart::Range(r) = self { r } else { panic!("Range is mandatory at this place") }
	}
}

/// Collection representing parsed input parts
type ParsedInput = Vec<InputPart>;

/// Parser of input string
///
/// String format is simple version of Bash expansion format.
/// For example, string `"Hello {1..3} world {4..6}!"` Is parsed as `[String("Hello "), Range(1, 3), String(" world "), Range(4, 6), String("!")]`
enum Parser {
	Empty,
	Str(String),
	FirstBrace,
	FirstNumber(u32),
	FirstDot(u32),
	SecondDot(u32),
	SecondNumber(InclusiveRangeU32),
}

impl Parser {
	/// Constructs new empty Parser
	fn new() -> Parser {
		Parser::Empty
	}

	/// Performs one step of parsing
	///
	/// Argument parsed is updated if new part was parsed.
	fn step(self, c: char, parsed: &mut ParsedInput) -> Parser {
		match self {
			Parser::Empty => match c {
				'{' => Parser::FirstBrace,
				'}' => panic!("Invalid syntax: closing curly brace doesn't match opening curly brace."),
				x => Parser::Str(x.to_string()),
			},
			Parser::Str(mut s) => match c {
				'{' => { parsed.push(InputPart::String(s)); Parser::FirstBrace },
				'}' => panic!("Invalid syntax: closing curly brace doesn't match opening curly brace."),
				x => { s.push(x); Parser::Str(s) },
			},
			Parser::FirstBrace => match c {
				digit @ '0' ... '9' => Parser::FirstNumber(digit.to_digit(10).unwrap()),
				x                   => panic!("Syntax error: number (0-9) expected, '{}' found", x),
			},
			Parser::FirstNumber(num) => match c {
				digit @ '0' ... '9' => Parser::FirstNumber(num * 10 + digit.to_digit(10).unwrap()),
				'.'                 => Parser::FirstDot(num),
				x                   => panic!("Syntax error: number ('0'-'9') or dot ('.') expected, '{}' found", x),
			},
			Parser::FirstDot(num) => match c {
				'.' => Parser::SecondDot(num),
				x   => panic!("Syntax error: dot ('.') expected, '{}' found", x),
			},
			Parser::SecondDot(num) => match c {
				digit @ '0' ... '9' => Parser::SecondNumber(InclusiveRangeU32 {begin: num, end: digit.to_digit(10).unwrap()}),
				x                   => panic!("Syntax error: number (0-9)  expected, '{}' found", x),
			},
			Parser::SecondNumber(r) => match c {
				digit @ '0' ... '9' => Parser::SecondNumber(InclusiveRangeU32 {begin: r.begin, end: r.end * 10 + digit.to_digit(10).unwrap()}),
				'}'                 => { parsed.push(InputPart::Range(r)); Parser::Empty }
				x                   => panic!("Syntax error: number (0-9) or closing brace ('}}') expected, '{}' found", x),
			}
		}
	}

	/// Finalizes parsing
	fn finish(self, parsed: &mut ParsedInput) {
		match self {
			Parser::Str(s)          => parsed.push(InputPart::String(s)),
			Parser::SecondNumber(r) => parsed.push(InputPart::Range(r)),
			_               => panic!("Syntax error: unfinished range"),
		}
	}

	/// Parses input string
	fn parse_input(input: &str) -> ParsedInput {
		let mut parsed_input = ParsedInput::new();
		let mut parser = Parser::new();

		for c in input.chars() {
			parser = parser.step(c, &mut parsed_input);
		}

		parser.finish(&mut parsed_input);

		parsed_input
	}
}

// ==================== Program logic ====================

/// Specifies how images should be placed
#[derive(PartialEq)]
enum Transposition {
	LeftRight,
	TopDown,
}

/// Opens images and extracts information from them
fn detect_tile_size_and_transposition(path1: &str, path2: &str) -> (u32, u32, Transposition) {
	let img1 = image::open(&Path::new(path1)).unwrap();
	let img2 = image::open(&Path::new(path2)).unwrap();

	let vertical_score = {
		let vline1 = PixelColumn::new(&img1, img1.width() - 1);
		let vline2 = PixelColumn::new(&img2, 0);

		line_match_score(&vline1, &vline2)
	};

	let horizontal_score = {
		let hline1 = PixelRow::new(&img1, img1.height() - 1);
		let hline2 = PixelRow::new(&img2, 0);

		line_match_score(&hline1, &hline2)
	};

	println!("Vertical score: {:?}", vertical_score);
	println!("Horizontal score: {:?}", horizontal_score);

	println!("Calculated configuration is:");
	let transposition = if vertical_score < horizontal_score {
		println!("[IMAGE1][IMAGE2]");
		Transposition::LeftRight
	} else {
		println!("[IMAGE1]");
		println!("[IMAGE2]");
		Transposition::TopDown
	};

	(img1.width(), img1.height(), transposition)
}

fn main() {
	let mut args = env::args();

	args.next();

	let input = args.next().expect("Usage: image_concat INPUT_IMAGE_PATTERN OUTPUT_IMAGE");
	let result_path = args.next().expect("Usage: image_concat INPUT_IMAGE_PATTERN OUTPUT_IMAGE");
	let mut parsed_input = Parser::parse_input(&input);
	
	if parsed_input.len() > 5 {
		panic!("Too many components! (Two separated ranges expected)");
	}

	let (suffix, second_range) = match parsed_input.pop().unwrap() {
		InputPart::String(s) => (s, parsed_input.pop().unwrap().take_mandatory_range()),
		InputPart::Range(r) => (String::new(), r),
	};

	let separator = parsed_input.pop().unwrap().take_mandatory_string();
	let first_range = parsed_input.pop().unwrap().take_mandatory_range();
	let prefix = parsed_input.pop().map_or_else(String::new, InputPart::take_mandatory_string);

	let corner_image_filename = format!("{}{}{}{}{}", prefix, first_range.begin, separator, second_range.begin, suffix);
	let other_image_filename = format!("{}{}{}{}{}", prefix, first_range.begin + 1, separator, second_range.begin, suffix);

	let (tile_width, tile_height, transposition) = detect_tile_size_and_transposition(&corner_image_filename, &other_image_filename);

	// Determine dimensions
	let (width, height) = if transposition == Transposition::LeftRight {
		((first_range.end - first_range.begin + 1) * tile_width, (second_range.end - second_range.begin + 1) * tile_height)
	} else {
		((second_range.end - second_range.begin + 1) * tile_width, (first_range.end - first_range.begin + 1) * tile_height)
	};

	let mut final_image = ImageBuffer::new(width, height);

	if transposition == Transposition::LeftRight {
		for x in first_range.iter() {
			for y in second_range.iter() {
				let image_filename = format!("{}{}{}{}{}", prefix, x, separator, y, suffix);
				let img = image::open(&Path::new(&image_filename)).unwrap();

				assert!(img.width() == tile_width);
				assert!(img.height() == tile_height);

				assert!(final_image.copy_from(&img, (x - first_range.begin) * tile_width, (y - second_range.begin) * tile_height));
			}
		}
	} else {
		for x in first_range.iter() {
			for y in second_range.iter() {
				let image_filename = format!("{}{}{}{}{}", prefix, x, separator, y, suffix);
				let img = image::open(&Path::new(&image_filename)).unwrap();

				assert!(img.width() == tile_width);
				assert!(img.height() == tile_height);

				assert!(final_image.copy_from(&img, (y - second_range.begin) * tile_width, (x - first_range.begin) * tile_height));
			}
		}
	}

	final_image.save(result_path).unwrap();
}
