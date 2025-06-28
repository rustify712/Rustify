use std::collections::HashMap;
use std::fmt;
use std::str::Chars;

/// Represents a JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    String(String),
    Number(String), // Keeping as String to handle various number formats.
    Object(JsonObject),
    Array(JsonArray),
    True,
    False,
    Null,
}

/// Represents a JSON object.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonObject {
    pub members: HashMap<String, JsonValue>,
}

/// Represents a JSON array.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonArray {
    pub elements: Vec<JsonValue>,
}

/// Parsing flags to customize parser behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsonParseOptions {
    pub allow_trailing_comma: bool,
    pub allow_unquoted_keys: bool,
    pub allow_global_object: bool,
    pub allow_equals_in_object: bool,
    pub allow_no_commas: bool,
    pub allow_c_style_comments: bool,
    pub allow_location_information: bool, // Not implemented in this version.
    pub allow_single_quoted_strings: bool,
    pub allow_hexadecimal_numbers: bool,
    pub allow_leading_plus_sign: bool,
    pub allow_leading_or_trailing_decimal_point: bool,
    pub allow_inf_and_nan: bool,
    pub allow_multi_line_strings: bool,
}

impl Default for JsonParseOptions {
    fn default() -> Self {
        Self {
            allow_trailing_comma: false,
            allow_unquoted_keys: false,
            allow_global_object: false,
            allow_equals_in_object: false,
            allow_no_commas: false,
            allow_c_style_comments: false,
            allow_location_information: false,
            allow_single_quoted_strings: false,
            allow_hexadecimal_numbers: false,
            allow_leading_plus_sign: false,
            allow_leading_or_trailing_decimal_point: false,
            allow_inf_and_nan: false,
            allow_multi_line_strings: false,
        }
    }
}

/// Represents a parsing error with detailed information.
#[derive(Debug, Clone, PartialEq)]
pub struct JsonParseError {
    pub error: JsonParseErrorKind,
    pub position: usize, // Character offset.
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JsonParseErrorKind {
    None,
    ExpectedCommaOrClosingBracket,
    ExpectedColon,
    ExpectedOpeningQuote,
    InvalidStringEscapeSequence,
    InvalidNumberFormat,
    InvalidValue,
    PrematureEndOfBuffer,
    InvalidString,
    AllocatorFailed,
    UnexpectedTrailingCharacters,
    Unknown,
}

impl fmt::Display for JsonParseErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use JsonParseErrorKind::*;
        let message = match self {
            None => "No error.",
            ExpectedCommaOrClosingBracket => "Expected comma or closing bracket.",
            ExpectedColon => "Expected colon.",
            ExpectedOpeningQuote => "Expected opening quote.",
            InvalidStringEscapeSequence => "Invalid string escape sequence.",
            InvalidNumberFormat => "Invalid number format.",
            InvalidValue => "Invalid value.",
            PrematureEndOfBuffer => "Premature end of buffer.",
            InvalidString => "Invalid string.",
            AllocatorFailed => "Allocator failed.",
            UnexpectedTrailingCharacters => "Unexpected trailing characters.",
            Unknown => "Unknown error.",
        };
        write!(f, "{}", message)
    }
}

impl std::error::Error for JsonParseErrorKind {}

/// Parses a JSON string into a JsonValue with given options.
/// Returns a Result containing the JsonValue or a JsonParseError.
pub fn parse_json(input: &str, options: JsonParseOptions) -> Result<JsonValue, JsonParseError> {
    let mut parser = JsonParser::new(input, options);
    let value = parser.parse_value()?;
    parser.skip_whitespace()?;
    if parser.current_char().is_some() {
        return Err(JsonParseError {
            error: JsonParseErrorKind::UnexpectedTrailingCharacters,
            position: parser.position,
            line: parser.line,
            column: parser.column,
        });
    }
    Ok(value)
}

/// Serializes a JsonValue into a minified JSON string.
pub fn to_minified_json(value: &JsonValue) -> String {
    let mut serializer = JsonSerializer::new();
    serializer.serialize_value(value, false, 0, "");
    serializer.output.clone() // 返回序列化后的字符串
}

/// Serializes a JsonValue into a pretty JSON string with given indentation.
pub fn to_pretty_json(value: &JsonValue, indent: &str) -> String {
    let mut serializer = JsonSerializer::new();
    serializer.serialize_value(value, true, 0, indent);
    serializer.output.clone() // 返回序列化后的字符串
}


/// Internal parser structure.
struct JsonParser<'a> {
    chars: Chars<'a>,
    current: Option<char>,
    position: usize, // Byte offset.
    line: usize,
    column: usize,
    options: JsonParseOptions,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str, options: JsonParseOptions) -> Self {
        let mut chars = input.chars();
        let current = chars.next();
        Self {
            chars,
            current,
            position: 0,
            line: 1,
            column: 1,
            options,
        }
    }

    fn advance(&mut self) {
        if let Some(c) = self.current {
            self.position += c.len_utf8();
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
        self.current = self.chars.next();
    }

    fn peek(&self) -> Option<char> {
        self.chars.clone().next()
    }

    fn current_char(&self) -> Option<char> {
        self.current
    }

    fn expect_char(&mut self, expected: char) -> Result<(), JsonParseError> {
        match self.current_char() {
            Some(c) if c == expected => {
                self.advance();
                Ok(())
            }
            Some(_) => Err(self.error(JsonParseErrorKind::Unknown)),
            None => Err(self.error(JsonParseErrorKind::PrematureEndOfBuffer)),
        }
    }

    fn error(&self, kind: JsonParseErrorKind) -> JsonParseError {
        JsonParseError {
            error: kind,
            position: self.position,
            line: self.line,
            column: self.column,
        }
    }

    fn skip_whitespace(&mut self) -> Result<(), JsonParseError> {
        while let Some(c) = self.current_char() {
            if c.is_whitespace() {
                self.advance();
            } else if self.options.allow_c_style_comments && c == '/' {
                if let Some(next) = self.peek() {
                    if next == '/' {
                        // Single-line comment
                        self.advance(); // Skip '/'
                        self.advance(); // Skip second '/'
                        while let Some(c) = self.current_char() {
                            if c == '\n' {
                                break;
                            }
                            self.advance();
                        }
                    } else if next == '*' {
                        // Multi-line comment
                        self.advance(); // Skip '/'
                        self.advance(); // Skip '*'
                        loop {
                            match self.current_char() {
                                Some('*') => {
                                    if self.peek() == Some('/') {
                                        self.advance(); // Skip '*'
                                        self.advance(); // Skip '/'
                                        break;
                                    } else {
                                        self.advance();
                                    }
                                }
                                Some(_) => self.advance(),
                                None => {
                                    return Err(self.error(JsonParseErrorKind::PrematureEndOfBuffer))
                                }
                            }
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    fn parse_value(&mut self) -> Result<JsonValue, JsonParseError> {
        self.skip_whitespace()?;
        match self.current_char() {
            Some('"') | Some('\'') if self.options.allow_single_quoted_strings => {
                self.parse_string()
            }
            Some('"') => self.parse_string(),
            Some('{') => self.parse_object(),
            Some('[') => self.parse_array(),
            Some('t') => self.parse_true(),
            Some('f') => self.parse_false(),
            Some('n') => self.parse_null(),
            Some(c) if c == '-' || c == '+' || c.is_digit(10) || c == '.' => {
                self.parse_number()
            }
            Some(_) => Err(self.error(JsonParseErrorKind::InvalidValue)),
            None => Err(self.error(JsonParseErrorKind::PrematureEndOfBuffer)),
        }
    }

    fn parse_string(&mut self) -> Result<JsonValue, JsonParseError> {
        let quote = self.current_char().unwrap();
        if quote != '"' && quote != '\'' {
            return Err(self.error(JsonParseErrorKind::ExpectedOpeningQuote));
        }
        self.advance(); // Skip opening quote
        let mut result = String::new();
        while let Some(c) = self.current_char() {
            if c == quote {
                self.advance(); // Skip closing quote
                return Ok(JsonValue::String(result));
            }
            if c == '\\' {
                self.advance(); // Skip backslash
                match self.current_char() {
                    Some('"') => {
                        result.push('"');
                        self.advance();
                    }
                    Some('\\') => {
                        result.push('\\');
                        self.advance();
                    }
                    Some('/') => {
                        result.push('/');
                        self.advance();
                    }
                    Some('b') => {
                        result.push('\u{0008}');
                        self.advance();
                    }
                    Some('f') => {
                        result.push('\u{000C}');
                        self.advance();
                    }
                    Some('n') => {
                        result.push('\n');
                        self.advance();
                    }
                    Some('r') => {
                        result.push('\r');
                        self.advance();
                    }
                    Some('t') => {
                        result.push('\t');
                        self.advance();
                    }
                    Some('u') => {
                        self.advance(); // Skip 'u'
                        let hex = self.consume_hex(4)?;
                        let codepoint =
                            u32::from_str_radix(&hex, 16).map_err(|_| {
                                self.error(JsonParseErrorKind::InvalidStringEscapeSequence)
                            })?;
                        if (0xD800..=0xDBFF).contains(&codepoint) {
                            // High surrogate, expect a low surrogate
                            if self.current_char() == Some('\\') && self.peek() == Some('u') {
                                self.advance(); // Skip '\\'
                                self.advance(); // Skip 'u'
                                let low_hex = self.consume_hex(4)?;
                                let low_codepoint =
                                    u32::from_str_radix(&low_hex, 16).map_err(|_| {
                                        self.error(JsonParseErrorKind::InvalidStringEscapeSequence)
                                    })?;
                                if (0xDC00..=0xDFFF).contains(&low_codepoint) {
                                    let combined = 0x10000
                                        + ((codepoint - 0xD800) << 10)
                                        + (low_codepoint - 0xDC00);
                                    if let Some(c) = std::char::from_u32(combined) {
                                        result.push(c);
                                    } else {
                                        return Err(self.error(
                                            JsonParseErrorKind::InvalidStringEscapeSequence,
                                        ));
                                    }
                                } else {
                                    return Err(self.error(
                                        JsonParseErrorKind::InvalidStringEscapeSequence,
                                    ));
                                }
                            } else {
                                return Err(self.error(
                                    JsonParseErrorKind::InvalidStringEscapeSequence,
                                ));
                            }
                        } else if (0xDC00..=0xDFFF).contains(&codepoint) {
                            // Unexpected low surrogate
                            return Err(self.error(
                                JsonParseErrorKind::InvalidStringEscapeSequence,
                            ));
                        } else {
                            if let Some(c) = std::char::from_u32(codepoint) {
                                result.push(c);
                            } else {
                                return Err(self.error(
                                    JsonParseErrorKind::InvalidStringEscapeSequence,
                                ));
                            }
                        }
                    }
                    Some(_) => {
                        return Err(self.error(JsonParseErrorKind::InvalidStringEscapeSequence));
                    }
                    None => {
                        return Err(self.error(JsonParseErrorKind::PrematureEndOfBuffer));
                    }
                }
            } else {
                if c == '\r' || c == '\n' {
                    if !self.options.allow_multi_line_strings {
                        return Err(self.error(JsonParseErrorKind::InvalidString));
                    }
                }
                result.push(c);
                self.advance();
            }
        }
        Err(self.error(JsonParseErrorKind::PrematureEndOfBuffer))
    }

    fn consume_hex(&mut self, count: usize) -> Result<String, JsonParseError> {
        let mut hex = String::new();
        for _ in 0..count {
            match self.current_char() {
                Some(c) if c.is_digit(16) => {
                    hex.push(c);
                    self.advance();
                }
                _ => return Err(self.error(JsonParseErrorKind::InvalidStringEscapeSequence)),
            }
        }
        Ok(hex)
    }

    fn parse_object(&mut self) -> Result<JsonValue, JsonParseError> {
        self.expect_char('{')?;
        self.skip_whitespace()?;
        let mut members = HashMap::new();
        let mut first = true;
        while let Some(c) = self.current_char() {
            if c == '}' {
                self.advance();
                return Ok(JsonValue::Object(JsonObject { members }));
            }
            if !first {
                if c == ',' {
                    self.advance();
                    if !self.options.allow_trailing_comma && self.current_char() == Some('}') {
                        return Err(self.error(JsonParseErrorKind::ExpectedColon));
                    }
                } else if self.options.allow_no_commas {
                    // No comma, continue
                } else {
                    return Err(self.error(JsonParseErrorKind::ExpectedCommaOrClosingBracket));
                }
            }
            first = false;
            self.skip_whitespace()?;
            // Parse key
            let key = if self.current_char() == Some('"')
                || (self.current_char() == Some('\'') && self.options.allow_single_quoted_strings)
            {
                match self.parse_string()? {
                    JsonValue::String(s) => s,
                    _ => unreachable!(),
                }
            } else if self.options.allow_unquoted_keys {
                self.parse_unquoted_key()?
            } else {
                return Err(self.error(JsonParseErrorKind::ExpectedOpeningQuote));
            };
            self.skip_whitespace()?;
            // Expect colon or equals
            match self.current_char() {
                Some(':') => self.advance(),
                Some('=') if self.options.allow_equals_in_object => self.advance(),
                Some(_) => return Err(self.error(JsonParseErrorKind::ExpectedColon)),
                None => return Err(self.error(JsonParseErrorKind::PrematureEndOfBuffer)),
            }
            self.skip_whitespace()?;
            // Parse value
            let value = self.parse_value()?;
            members.insert(key, value);
            self.skip_whitespace()?;
        }
        Err(self.error(JsonParseErrorKind::PrematureEndOfBuffer))
    }

    fn parse_unquoted_key(&mut self) -> Result<String, JsonParseError> {
        let mut key = String::new();
        while let Some(c) = self.current_char() {
            if is_valid_unquoted_key_char(c) {
                key.push(c);
                self.advance();
            } else {
                break;
            }
        }
        if key.is_empty() {
            return Err(self.error(JsonParseErrorKind::InvalidValue));
        }
        Ok(key)
    }

    fn parse_array(&mut self) -> Result<JsonValue, JsonParseError> {
        self.expect_char('[')?;
        self.skip_whitespace()?;
        let mut elements = Vec::new();
        let mut first = true;
        while let Some(c) = self.current_char() {
            if c == ']' {
                self.advance();
                return Ok(JsonValue::Array(JsonArray { elements }));
            }
            if !first {
                if c == ',' {
                    self.advance();
                    if !self.options.allow_trailing_comma && self.current_char() == Some(']') {
                        return Err(self.error(JsonParseErrorKind::ExpectedColon));
                    }
                } else if self.options.allow_no_commas {
                    // No comma, continue
                } else {
                    return Err(self.error(JsonParseErrorKind::ExpectedCommaOrClosingBracket));
                }
            }
            first = false;
            self.skip_whitespace()?;
            let value = self.parse_value()?;
            elements.push(value);
            self.skip_whitespace()?;
        }
        Err(self.error(JsonParseErrorKind::PrematureEndOfBuffer))
    }

    fn parse_true(&mut self) -> Result<JsonValue, JsonParseError> {
        if self.match_literal("true") {
            Ok(JsonValue::True)
        } else {
            Err(self.error(JsonParseErrorKind::InvalidValue))
        }
    }

    fn parse_false(&mut self) -> Result<JsonValue, JsonParseError> {
        if self.match_literal("false") {
            Ok(JsonValue::False)
        } else {
            Err(self.error(JsonParseErrorKind::InvalidValue))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, JsonParseError> {
        if self.match_literal("null") {
            Ok(JsonValue::Null)
        } else {
            Err(self.error(JsonParseErrorKind::InvalidValue))
        }
    }

    fn match_literal(&mut self, literal: &str) -> bool {
        let mut iter = self.chars.clone();
        for expected_char in literal.chars() {
            match iter.next() {
                Some(c) if c == expected_char => {}
                _ => return false,
            }
        }
        for _ in 0..literal.len() {
            self.advance();
        }
        true
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonParseError> {
        // Removed the unused `start_pos` variable
        let mut number = String::new();

        // Optional leading sign
        if let Some(c) = self.current_char() {
            if c == '-' {
                number.push(c);
                self.advance();
            } else if c == '+' && self.options.allow_leading_plus_sign {
                self.advance(); // Skip '+', but don't include it
            }
        }

        // Hexadecimal
        if self.options.allow_hexadecimal_numbers {
            if self.current_char() == Some('0')
                && (self.peek() == Some('x') || self.peek() == Some('X'))
            {
                number.push('0');
                self.advance(); // Skip '0'
                if let Some(c) = self.current_char() {
                    if c == 'x' || c == 'X' {
                        number.push(c);
                        self.advance(); // Skip 'x' or 'X'
                        while let Some(c) = self.current_char() {
                            if c.is_digit(16) {
                                number.push(c);
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        return Ok(JsonValue::Number(number));
                    }
                }
            }
        }

        // Integer part
        while let Some(c) = self.current_char() {
            if c.is_digit(10) {
                number.push(c);
                self.advance();
            } else {
                break;
            }
        }

        // Fractional part
        if self.current_char() == Some('.') {
            if !self.options.allow_leading_or_trailing_decimal_point && number.is_empty() {
                return Err(self.error(JsonParseErrorKind::InvalidNumberFormat));
            }
            number.push('.');
            self.advance();
            let mut has_digits = false;
            while let Some(c) = self.current_char() {
                if c.is_digit(10) {
                    has_digits = true;
                    number.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
            if !has_digits {
                if !self.options.allow_leading_or_trailing_decimal_point {
                    return Err(self.error(JsonParseErrorKind::InvalidNumberFormat));
                }
            }
        }

        // Exponent part
        if let Some(c) = self.current_char() {
            if c == 'e' || c == 'E' {
                number.push(c);
                self.advance();
                if let Some(c) = self.current_char() {
                    if c == '-' || c == '+' {
                        number.push(c);
                        self.advance();
                    }
                }
                let mut has_digits = false;
                while let Some(c) = self.current_char() {
                    if c.is_digit(10) {
                        has_digits = true;
                        number.push(c);
                        self.advance();
                    } else {
                        break;
                    }
                }
                if !has_digits {
                    return Err(self.error(JsonParseErrorKind::InvalidNumberFormat));
                }
            }
        }

        // Special values: Infinity and NaN
        if self.options.allow_inf_and_nan {
            if self.match_literal("Infinity") {
                number = "Infinity".to_string();
                return Ok(JsonValue::Number(number));
            } else if self.match_literal("NaN") {
                number = "NaN".to_string();
                return Ok(JsonValue::Number(number));
            }
        }

        // Check for trailing decimal point
        if self.current_char() == Some('.') {
            if self.options.allow_leading_or_trailing_decimal_point {
                number.push('.');
                self.advance();
                number.push('0'); // Append '0' to fix the trailing decimal point
            } else {
                return Err(self.error(JsonParseErrorKind::InvalidNumberFormat));
            }
        }

        if number.is_empty() {
            return Err(self.error(JsonParseErrorKind::InvalidNumberFormat));
        }

        Ok(JsonValue::Number(number))
    }
}

/// Checks if a character is valid for an unquoted key.
/// According to JSON specifications, keys must be strings. However, some parsers allow
/// unquoted keys with specific character sets. Here, we'll define a simple set:
/// Alphanumeric characters and underscores.
fn is_valid_unquoted_key_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Internal serializer structure.
struct JsonSerializer {
    output: String,
}

impl JsonSerializer {
    fn new() -> Self {
        Self {
            output: String::new(),
        }
    }

    /// Serializes a JsonValue.
    ///
    /// # Arguments
    ///
    /// * `value` - The JsonValue to serialize.
    /// * `pretty` - Whether to format the JSON in a pretty (indented) manner.
    /// * `depth` - Current depth for indentation.
    /// * `indent` - String used for each indentation level.
    fn serialize_value(
        &mut self,
        value: &JsonValue,
        pretty: bool,
        depth: usize,
        indent: &str,
    ) -> &Self {
        match value {
            JsonValue::String(s) => self.serialize_string(s),
            JsonValue::Number(n) => self.serialize_number(n),
            JsonValue::Object(obj) => self.serialize_object(obj, pretty, depth, indent),
            JsonValue::Array(arr) => self.serialize_array(arr, pretty, depth, indent),
            JsonValue::True => {
                self.output.push_str("true");
                self
            }
            JsonValue::False => {
                self.output.push_str("false");
                self
            }
            JsonValue::Null => {
                self.output.push_str("null");
                self
            }
        }
    }

    fn serialize_string(&mut self, s: &str) -> &Self {
        self.output.push('"');
        for c in s.chars() {
            match c {
                '"' => {
                    self.output.push('\\');
                    self.output.push('"');
                }
                '\\' => {
                    self.output.push('\\');
                    self.output.push('\\');
                }
                '\u{0008}' => {
                    self.output.push('\\');
                    self.output.push('b');
                }
                '\u{000C}' => {
                    self.output.push('\\');
                    self.output.push('f');
                }
                '\n' => {
                    self.output.push('\\');
                    self.output.push('n');
                }
                '\r' => {
                    self.output.push('\\');
                    self.output.push('r');
                }
                '\t' => {
                    self.output.push('\\');
                    self.output.push('t');
                }
                _ => {
                    self.output.push(c);
                }
            }
        }
        self.output.push('"');
        self
    }

    fn serialize_number(&mut self, n: &str) -> &Self {
        self.output.push_str(n);
        self
    }

    fn serialize_object(
        &mut self,
        obj: &JsonObject,
        pretty: bool,
        depth: usize,
        indent: &str,
    ) -> &Self {
        self.output.push('{');
        if !obj.members.is_empty() {
            if pretty {
                self.output.push('\n');
            }
            let mut first = true;
            for (k, v) in &obj.members {
                if !first {
                    self.output.push(',');
                    if pretty {
                        self.output.push('\n');
                    }
                }
                first = false;
                if pretty {
                    for _ in 0..depth + 1 {
                        self.output.push_str(indent);
                    }
                }
                self.serialize_string(k);
                if pretty {
                    self.output.push_str(": ");
                } else {
                    self.output.push(':');
                }
                self.serialize_value(v, pretty, depth + 1, indent);
            }
            if pretty {
                self.output.push('\n');
                for _ in 0..depth {
                    self.output.push_str(indent);
                }
            }
        }
        self.output.push('}');
        self
    }

    fn serialize_array(
        &mut self,
        arr: &JsonArray,
        pretty: bool,
        depth: usize,
        indent: &str,
    ) -> &Self {
        self.output.push('[');
        if !arr.elements.is_empty() {
            if pretty {
                self.output.push('\n');
            }
            let mut first = true;
            for v in &arr.elements {
                if !first {
                    self.output.push(',');
                    if pretty {
                        self.output.push('\n');
                    }
                }
                first = false;
                if pretty {
                    for _ in 0..depth + 1 {
                        self.output.push_str(indent);
                    }
                }
                self.serialize_value(v, pretty, depth + 1, indent);
            }
            if pretty {
                self.output.push('\n');
                for _ in 0..depth {
                    self.output.push_str(indent);
                }
            }
        }
        self.output.push(']');
        self
    }
}