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
use std::ops::Deref;

// ==================== Pixels ====================

/// Represents generic sequence of pixels
trait PixelLine {
	/// Type of Pixel - as in GenericImage
	type Pixel: Pixel;

	/// Returns length of the sequence
	fn len(&self) -> u32;

	/// Returns copy of the pixel at given position
	fn get_pixel(&self, index: u32) -> Self::Pixel;

    /// Whether the pixel line is empty
    fn is_empty(&self) -> bool;
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
        let w = img.width();
        if column >= w {
            Self::new(img, w - 1)
        } else {
            PixelColumn {column_index: column, img: img}
        }
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

    fn is_empty(&self) -> bool {
        self.img.height() == 0
    }
}

impl<'a, I> PixelLine for PixelRow<'a, I> where I: GenericImage {
	type Pixel = I::Pixel;
    fn is_empty(&self) -> bool {
        self.img.width() == 0
    }

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
#[derive(Debug)]
#[derive(Clone)]
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

impl Default for Parser {
    fn default() -> Parser {
        Parser::new()
    }
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
			Parser::Str(s) => parsed.push(InputPart::String(s)),
			Parser::Empty  => {},
			_              => panic!("Syntax error: unfinished range"),
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

/// Vec guaranteed to be sorted (newtype pattern)
struct SortedVec<T>(Vec<T>) where T: Ord;

impl<T> SortedVec<T> where T: Ord {
	/// Creates new `SortedVec` from unsorted `Vec`
	fn new(mut vec: Vec<T>) -> SortedVec<T> {
		vec.sort();
		SortedVec::<T>(vec)
	}

	/// Gives up sorted property to allow mutations.
	fn to_vec(self) -> Vec<T> {
		self.0
	}

	//Allow some safe mutations
	fn remove(&mut self, index: usize) -> T {
		self.0.remove(index)
	}

	fn pop(&mut self) -> Option<T> {
		 self.0.pop()
	}

	fn dedup(&mut self) {
		self.0.dedup()
	}
}

impl<T> Deref for SortedVec<T> where T: Ord {
	type Target = Vec<T>;

	// Only immutable deref provided, so no changes are allowed,
	// which means nothing can break sorted guarantee.
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

#[derive(Debug)]
#[derive(Clone)]
enum StrDiff {
	Same(usize),
	Diff(usize),
}

type StrDiffs = Vec<StrDiff>;

/// Represents state of analysis of different numbers
enum AnalyzerState {
    Empty,
    SamePartWithoutDigit(usize),
    SamePartWithDigit(usize, usize),
    DifferentPart(usize),
}

impl SortedVec<String> {
	fn analyze_different_numbers(& self) -> Vec<StrDiffs> {
		let mut result = Vec::<StrDiffs>::new();
		let mut iter = (*self).iter();
		let mut last = if let Some(s) = iter.next() {
			s
		} else {
			return result
		};

		for str in iter {
			if last.len() != str.len() {
				panic!("File names are not of same length");
			}

            let mut tmp_res = StrDiffs::new();
            let mut byte_index = 0usize;
            let mut state = AnalyzerState::Empty;
            let striter = last.chars().zip(str.chars());

			for (c1, c2) in striter {
                state = match state {
                    AnalyzerState::Empty => {
                        if c1.is_digit(10) && c2.is_digit(10) {
                            if c1 == c2 {
                                AnalyzerState::SamePartWithDigit(byte_index, byte_index)
                            } else {
                                AnalyzerState::DifferentPart(byte_index)
                            }
                        } else {
                            if c1 != c2 {
                                panic!("Strings differ in something that is not digit");
                            } 

                            AnalyzerState::SamePartWithoutDigit(byte_index)
                        }
                    },
                    AnalyzerState::SamePartWithoutDigit(begin) => {
                        if c1.is_digit(10) && c2.is_digit(10) {
                            if c1 == c2 {
                                AnalyzerState::SamePartWithDigit(begin, byte_index)
                            } else {
                                tmp_res.push(StrDiff::Same(byte_index - begin));
                                AnalyzerState::DifferentPart(byte_index)
                            }
                        } else {
                            if c1 != c2 {
                                panic!("Strings differ in something that is not digit");
                            } 

                            AnalyzerState::SamePartWithoutDigit(begin)
                        }
                    },
                    AnalyzerState::SamePartWithDigit(begin, first_digit) => {
                        if c1.is_digit(10) && c2.is_digit(10) {
                            if c1 == c2 {
                                AnalyzerState::SamePartWithDigit(begin, first_digit)
                            } else {
                                if begin != first_digit {
                                    tmp_res.push(StrDiff::Same(first_digit - begin));
                                }

                                AnalyzerState::DifferentPart(first_digit)
                            }
                        } else {
                            if c1 != c2 {
                                panic!("Strings differ in something that is not digit");
                            }

                            AnalyzerState::SamePartWithoutDigit(begin)
                        }
                    },
                    AnalyzerState::DifferentPart(first_digit) => {
                        if c1.is_digit(10) && c2.is_digit(10) {
                            AnalyzerState::DifferentPart(first_digit)
                        } else {
                            if c1 != c2 {
                                panic!("Strings differ in something that is not digit");
                            }

                            tmp_res.push(StrDiff::Diff(byte_index - first_digit));
                            AnalyzerState::SamePartWithoutDigit(byte_index)
                        }
                    }
                };
                byte_index += c1.len_utf8();
            }

            match state {
                    AnalyzerState::Empty => {},
                    AnalyzerState::SamePartWithoutDigit(begin) => {
                        tmp_res.push(StrDiff::Same(byte_index - begin));
                    },
                    AnalyzerState::SamePartWithDigit(begin, _) => {
                        tmp_res.push(StrDiff::Same(byte_index - begin));
                    },
                    AnalyzerState::DifferentPart(first_digit) => {
                        tmp_res.push(StrDiff::Diff(byte_index - first_digit));
                    },
            }

			result.push(tmp_res);
			last = str;
		}

		result
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

struct InputPattern {
	prefix: String,
	first_range: InclusiveRangeU32,
	separator: String,
	second_range: InclusiveRangeU32,
	suffix: String,
}

fn consume_strdiff_iter_upto_len<'a, T>(max_len: usize, input_iter: T, res: &mut StrDiffs) where T: IntoIterator<Item=&'a StrDiff> {
    let mut iter = input_iter.into_iter();
    let mut pushed_len = 0;
    while pushed_len < max_len {
        let item = iter.next().unwrap();
        res.push(item.clone());
        pushed_len += match *item {
            StrDiff::Same(l) => l,
            StrDiff::Diff(l) => l,
        };
    }
}

enum MaybeOwnedStrDiffs<'a> {
    Owned(StrDiffs),
    Borrowed(&'a StrDiffs),
}

impl<'a> Deref for MaybeOwnedStrDiffs<'a> {
    type Target = StrDiffs;

    fn deref(&self) -> &Self::Target {
        match *self {
            MaybeOwnedStrDiffs::Owned(ref owned) => owned,
            MaybeOwnedStrDiffs::Borrowed(borrowed) => borrowed,
        }
    }
}

fn collapse_diff_patterns(patterns: &Vec<StrDiffs>) -> StrDiffs {
    let mut iter = patterns.iter();
    let mut res = MaybeOwnedStrDiffs::Borrowed(iter.next().unwrap());

    for pattern in iter {
        if pattern.len() != res.len() {
            let mut tmp = StrDiffs::new();
            {
                let mut pat_iter = res.iter();
                let mut cur_iter = pattern.iter();
                loop {
                    if let Some(pat) = pat_iter.next() {
                        if let Some(cur) = cur_iter.next() {
                            let pat_len = match *pat {
                                StrDiff::Same(l) => l,
                                StrDiff::Diff(l) => l,
                            };
                            let cur_len = match *cur {
                                StrDiff::Same(l) => l,
                                StrDiff::Diff(l) => l,
                            };

                            if pat_len > cur_len {
                                tmp.push(if let StrDiff::Same(_) = *cur { StrDiff::Same(cur_len) } else { StrDiff::Diff(cur_len) });
                                consume_strdiff_iter_upto_len(pat_len - cur_len, &mut cur_iter, &mut tmp);
                            } else {
                                tmp.push(if let StrDiff::Same(_) = *pat { StrDiff::Same(pat_len) } else { StrDiff::Diff(pat_len) });
                                consume_strdiff_iter_upto_len(cur_len - pat_len, &mut pat_iter, &mut tmp);
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }

            res = MaybeOwnedStrDiffs::Owned(tmp);
        }
    }

    match res {
        MaybeOwnedStrDiffs::Owned(owned) => owned,
        MaybeOwnedStrDiffs::Borrowed(borrowed) => borrowed.clone(),
    }
}

fn parse_ranges(pattern: &StrDiffs, path: &str) -> Vec<InclusiveRangeU32> {
    let mut res = Vec::<InclusiveRangeU32>::new();
    let mut pos = 0usize;

    for str_diff in pattern {
        pos += match *str_diff {
            StrDiff::Same(same) => {
                same
            },
            StrDiff::Diff(different) => {
                let ref slice = path[pos..pos + different];
                let number = slice.parse::<u32>().unwrap();
                //let number = path[pos..pos + different].parse::<u32>().unwrap();
                res.push(InclusiveRangeU32 { begin: number, end: number });
                different
            },
        }
    }

    res
}

fn update_ranges(pattern: &StrDiffs, path: &str, ranges: &mut Vec<InclusiveRangeU32>) {
    let mut iter = ranges.iter_mut();
    let mut pos = 0usize;

    for str_diff in pattern {
        pos += match *str_diff {
            StrDiff::Same(same) => {
                same
            },
            StrDiff::Diff(different) => {
                let ref slice = path[pos..pos + different];
                let number = slice.parse::<u32>().unwrap();
                let mut range = iter.next().unwrap();

                if number < range.begin { range.begin = number; }
                if number > range.end { range.end = number; }

                different
            },
        }
    }
}

fn parse_multi_arg(all_args: env::Args) -> (InputPattern, String) {
	let mut args = all_args.collect::<Vec<String>>();
	let last = args.pop().unwrap();

	let sorted_args = SortedVec::<String>::new(args);

	let differences = sorted_args.analyze_different_numbers();

    let collapsed_pattern = collapse_diff_patterns(&differences);
    println!("Collapsed: {:#?}", collapsed_pattern);

    let mut arg_iter = sorted_args.iter();
    let mut ranges = parse_ranges(&collapsed_pattern, &arg_iter.next().unwrap());

    for arg in arg_iter {
        update_ranges(&collapsed_pattern, &arg, &mut ranges);
    }
    println!("Ranges: {:#?}", ranges);

    if ranges.len() != 2 {
        panic!("Can not understand file names - they should have two differing numbers in their paths");
    }

    let arg0 = &sorted_args[0];

    let mut range_iter = ranges.iter();
    let mut pattern_iter = collapsed_pattern.iter();
    let mut pos = 0usize;

    let (prefix, first_range) = {
        match *pattern_iter.next().unwrap() {
            StrDiff::Same(l) => {
                let pref = &arg0[pos..pos + l];
                pos += l;
                if let StrDiff::Diff(d) = *pattern_iter.next().unwrap() {
                    pos += d;
                } else {
                    panic!("This error can never occur");
                }

                (pref, range_iter.next().unwrap().clone())
            },
            StrDiff::Diff(d) => {
                pos += d;
                ("", range_iter.next().unwrap().clone())
            },
        }
    };

    let sep_len = if let StrDiff::Same(l) = *pattern_iter.next().unwrap() { l } else { panic!("This error can never occur") };
    let separator = &arg0[pos..pos + sep_len];
    pos += sep_len;
    let second_range = (*range_iter.next().unwrap()).clone();
    pos += if let StrDiff::Diff(l) = *pattern_iter.next().unwrap() { l } else { panic!("This error can never occur") };
    let suffix = if let Some(last) = pattern_iter.next() {
        let suff_len = if let StrDiff::Same(l) = *last { l } else { panic!("This error can never occur") };
        &arg0[pos..pos + suff_len]
    } else {
        ""
    };


    (
        InputPattern {
            prefix: prefix.to_string(),
            first_range: first_range,
            separator: separator.to_string(),
            second_range: second_range,
            suffix: suffix.to_string(),
        },
        last
    )
}

fn parse_two_arg(mut args: env::Args) -> (InputPattern, String) {
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

	(
		InputPattern {
			prefix: prefix,
			first_range: first_range,
			separator: separator,
			second_range: second_range,
			suffix: suffix,
		},
		result_path
	)
}

fn main() {
	let mut args = env::args();

	args.next();

	let (pattern, result_path) = if args.len() > 2 {
		parse_multi_arg(args)
	} else {
		parse_two_arg(args)
	};

	let InputPattern { prefix, first_range, separator, second_range, suffix} = pattern;

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
