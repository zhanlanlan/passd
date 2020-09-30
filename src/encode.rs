use core::convert::TryInto;
use std::{error, fmt};

pub fn encode<T: AsRef<[u8]>>(input: T) -> String {
    let mut input = input.as_ref().to_vec();
    input.reverse();
    let mut buf = match encoded_size(input.len(), STANDARD) {
        Some(n) => vec![0; n],
        None => panic!("integer overflow when calculating buffer size"),
    };

    encode_with_padding(input.as_ref(), STANDARD, buf.len(), &mut buf[..]);

    String::from_utf8(buf).expect("Invalid UTF8")
}

pub fn decode<T: AsRef<[u8]>>(input: T) -> Result<Vec<u8>, DecodeError> {
    let mut buffer = Vec::<u8>::with_capacity(input.as_ref().len() * 4 / 3);

    decode_config_buf(input, STANDARD, &mut buffer).map(|_| {
        buffer.reverse();
        return buffer;
    })
}

///Decode from string reference as octets.
///Writes into the supplied buffer to avoid allocation.
///Returns a Result containing an empty tuple, aka ().
///
///# Example
///
///```rust
///extern crate base64;
///
///fn main() {
///    let mut buffer = Vec::<u8>::new();
///    base64::decode_config_buf("aGVsbG8gd29ybGR+Cg==", base64::STANDARD, &mut buffer).unwrap();
///    println!("{:?}", buffer);
///
///    buffer.clear();
///
///    base64::decode_config_buf("aGVsbG8gaW50ZXJuZXR-Cg==", base64::URL_SAFE, &mut buffer)
///        .unwrap();
///    println!("{:?}", buffer);
///}
///```
fn decode_config_buf<T: AsRef<[u8]>>(
    input: T,
    config: Config,
    buffer: &mut Vec<u8>,
) -> Result<(), DecodeError> {
    let input_bytes = input.as_ref();

    let starting_output_len = buffer.len();

    let num_chunks = num_chunks(input_bytes);
    let decoded_len_estimate = num_chunks
        .checked_mul(DECODED_CHUNK_LEN)
        .and_then(|p| p.checked_add(starting_output_len))
        .expect("Overflow when calculating output buffer length");
    buffer.resize(decoded_len_estimate, 0);

    let bytes_written;
    {
        let buffer_slice = &mut buffer.as_mut_slice()[starting_output_len..];
        bytes_written = decode_helper(input_bytes, num_chunks, config, buffer_slice)?;
    }

    buffer.truncate(starting_output_len + bytes_written);

    Ok(())
}

/// Helper to avoid duplicating num_chunks calculation, which is costly on short inputs.
/// Returns the number of bytes written, or an error.
// We're on the fragile edge of compiler heuristics here. If this is not inlined, slow. If this is
// inlined(always), a different slow. plain ol' inline makes the benchmarks happiest at the moment,
// but this is fragile and the best setting changes with only minor code modifications.
#[inline]
fn decode_helper(
    input: &[u8],
    num_chunks: usize,
    config: Config,
    output: &mut [u8],
) -> Result<usize, DecodeError> {
    let char_set = config.char_set;
    let decode_table = char_set.decode_table();

    let remainder_len = input.len() % INPUT_CHUNK_LEN;

    // Because the fast decode loop writes in groups of 8 bytes (unrolled to
    // CHUNKS_PER_FAST_LOOP_BLOCK times 8 bytes, where possible) and outputs 8 bytes at a time (of
    // which only 6 are valid data), we need to be sure that we stop using the fast decode loop
    // soon enough that there will always be 2 more bytes of valid data written after that loop.
    let trailing_bytes_to_skip = match remainder_len {
        // if input is a multiple of the chunk size, ignore the last chunk as it may have padding,
        // and the fast decode logic cannot handle padding
        0 => INPUT_CHUNK_LEN,
        // 1 and 5 trailing bytes are illegal: can't decode 6 bits of input into a byte
        1 | 5 => return Err(DecodeError::InvalidLength),
        // This will decode to one output byte, which isn't enough to overwrite the 2 extra bytes
        // written by the fast decode loop. So, we have to ignore both these 2 bytes and the
        // previous chunk.
        2 => INPUT_CHUNK_LEN + 2,
        // If this is 3 unpadded chars, then it would actually decode to 2 bytes. However, if this
        // is an erroneous 2 chars + 1 pad char that would decode to 1 byte, then it should fail
        // with an error, not panic from going past the bounds of the output slice, so we let it
        // use stage 3 + 4.
        3 => INPUT_CHUNK_LEN + 3,
        // This can also decode to one output byte because it may be 2 input chars + 2 padding
        // chars, which would decode to 1 byte.
        4 => INPUT_CHUNK_LEN + 4,
        // Everything else is a legal decode len (given that we don't require padding), and will
        // decode to at least 2 bytes of output.
        _ => remainder_len,
    };

    // rounded up to include partial chunks
    let mut remaining_chunks = num_chunks;

    let mut input_index = 0;
    let mut output_index = 0;

    {
        let length_of_fast_decode_chunks = input.len().saturating_sub(trailing_bytes_to_skip);

        // Fast loop, stage 1
        // manual unroll to CHUNKS_PER_FAST_LOOP_BLOCK of u64s to amortize slice bounds checks
        if let Some(max_start_index) = length_of_fast_decode_chunks.checked_sub(INPUT_BLOCK_LEN) {
            while input_index <= max_start_index {
                let input_slice = &input[input_index..(input_index + INPUT_BLOCK_LEN)];
                let output_slice = &mut output[output_index..(output_index + DECODED_BLOCK_LEN)];

                decode_chunk(
                    &input_slice[0..],
                    input_index,
                    decode_table,
                    &mut output_slice[0..],
                )?;
                decode_chunk(
                    &input_slice[8..],
                    input_index + 8,
                    decode_table,
                    &mut output_slice[6..],
                )?;
                decode_chunk(
                    &input_slice[16..],
                    input_index + 16,
                    decode_table,
                    &mut output_slice[12..],
                )?;
                decode_chunk(
                    &input_slice[24..],
                    input_index + 24,
                    decode_table,
                    &mut output_slice[18..],
                )?;

                input_index += INPUT_BLOCK_LEN;
                output_index += DECODED_BLOCK_LEN - DECODED_CHUNK_SUFFIX;
                remaining_chunks -= CHUNKS_PER_FAST_LOOP_BLOCK;
            }
        }

        // Fast loop, stage 2 (aka still pretty fast loop)
        // 8 bytes at a time for whatever we didn't do in stage 1.
        if let Some(max_start_index) = length_of_fast_decode_chunks.checked_sub(INPUT_CHUNK_LEN) {
            while input_index < max_start_index {
                decode_chunk(
                    &input[input_index..(input_index + INPUT_CHUNK_LEN)],
                    input_index,
                    decode_table,
                    &mut output
                        [output_index..(output_index + DECODED_CHUNK_LEN + DECODED_CHUNK_SUFFIX)],
                )?;

                output_index += DECODED_CHUNK_LEN;
                input_index += INPUT_CHUNK_LEN;
                remaining_chunks -= 1;
            }
        }
    }

    // Stage 3
    // If input length was such that a chunk had to be deferred until after the fast loop
    // because decoding it would have produced 2 trailing bytes that wouldn't then be
    // overwritten, we decode that chunk here. This way is slower but doesn't write the 2
    // trailing bytes.
    // However, we still need to avoid the last chunk (partial or complete) because it could
    // have padding, so we always do 1 fewer to avoid the last chunk.
    for _ in 1..remaining_chunks {
        decode_chunk_precise(
            &input[input_index..],
            input_index,
            decode_table,
            &mut output[output_index..(output_index + DECODED_CHUNK_LEN)],
        )?;

        input_index += INPUT_CHUNK_LEN;
        output_index += DECODED_CHUNK_LEN;
    }

    // always have one more (possibly partial) block of 8 input
    debug_assert!(input.len() - input_index > 1 || input.is_empty());
    debug_assert!(input.len() - input_index <= 8);

    // Stage 4
    // Finally, decode any leftovers that aren't a complete input block of 8 bytes.
    // Use a u64 as a stack-resident 8 byte buffer.
    let mut leftover_bits: u64 = 0;
    let mut morsels_in_leftover = 0;
    let mut padding_bytes = 0;
    let mut first_padding_index: usize = 0;
    let mut last_symbol = 0_u8;
    let start_of_leftovers = input_index;
    for (i, b) in input[start_of_leftovers..].iter().enumerate() {
        // '=' padding
        if *b == 0x3D {
            // There can be bad padding in a few ways:
            // 1 - Padding with non-padding characters after it
            // 2 - Padding after zero or one non-padding characters before it
            //     in the current quad.
            // 3 - More than two characters of padding. If 3 or 4 padding chars
            //     are in the same quad, that implies it will be caught by #2.
            //     If it spreads from one quad to another, it will be caught by
            //     #2 in the second quad.

            if i % 4 < 2 {
                // Check for case #2.
                let bad_padding_index = start_of_leftovers
                    + if padding_bytes > 0 {
                        // If we've already seen padding, report the first padding index.
                        // This is to be consistent with the faster logic above: it will report an
                        // error on the first padding character (since it doesn't expect to see
                        // anything but actual encoded data).
                        first_padding_index
                    } else {
                        // haven't seen padding before, just use where we are now
                        i
                    };
                return Err(DecodeError::InvalidByte(bad_padding_index, *b));
            }

            if padding_bytes == 0 {
                first_padding_index = i;
            }

            padding_bytes += 1;
            continue;
        }

        // Check for case #1.
        // To make '=' handling consistent with the main loop, don't allow
        // non-suffix '=' in trailing chunk either. Report error as first
        // erroneous padding.
        if padding_bytes > 0 {
            return Err(DecodeError::InvalidByte(
                start_of_leftovers + first_padding_index,
                0x3D,
            ));
        }
        last_symbol = *b;

        // can use up to 8 * 6 = 48 bits of the u64, if last chunk has no padding.
        // To minimize shifts, pack the leftovers from left to right.
        let shift = 64 - (morsels_in_leftover + 1) * 6;
        // tables are all 256 elements, lookup with a u8 index always succeeds
        let morsel = decode_table[*b as usize];
        if morsel == INVALID_VALUE {
            return Err(DecodeError::InvalidByte(start_of_leftovers + i, *b));
        }

        leftover_bits |= (morsel as u64) << shift;
        morsels_in_leftover += 1;
    }

    let leftover_bits_ready_to_append = match morsels_in_leftover {
        0 => 0,
        2 => 8,
        3 => 16,
        4 => 24,
        6 => 32,
        7 => 40,
        8 => 48,
        _ => unreachable!(
            "Impossible: must only have 0 to 8 input bytes in last chunk, with no invalid lengths"
        ),
    };

    // if there are bits set outside the bits we care about, last symbol encodes trailing bits that
    // will not be included in the output
    let mask = !0 >> leftover_bits_ready_to_append;
    if !config.decode_allow_trailing_bits && (leftover_bits & mask) != 0 {
        // last morsel is at `morsels_in_leftover` - 1
        return Err(DecodeError::InvalidLastSymbol(
            start_of_leftovers + morsels_in_leftover - 1,
            last_symbol,
        ));
    }

    let mut leftover_bits_appended_to_buf = 0;
    while leftover_bits_appended_to_buf < leftover_bits_ready_to_append {
        // `as` simply truncates the higher bits, which is what we want here
        let selected_bits = (leftover_bits >> (56 - leftover_bits_appended_to_buf)) as u8;
        output[output_index] = selected_bits;
        output_index += 1;

        leftover_bits_appended_to_buf += 8;
    }

    Ok(output_index)
}

/// Decode 8 bytes of input into 6 bytes of output. 8 bytes of output will be written, but only the
/// first 6 of those contain meaningful data.
///
/// `input` is the bytes to decode, of which the first 8 bytes will be processed.
/// `index_at_start_of_input` is the offset in the overall input (used for reporting errors
/// accurately)
/// `decode_table` is the lookup table for the particular base64 alphabet.
/// `output` will have its first 8 bytes overwritten, of which only the first 6 are valid decoded
/// data.
// yes, really inline (worth 30-50% speedup)
#[inline(always)]
fn decode_chunk(
    input: &[u8],
    index_at_start_of_input: usize,
    decode_table: &[u8; 256],
    output: &mut [u8],
) -> Result<(), DecodeError> {
    let mut accum: u64;

    let morsel = decode_table[input[0] as usize];
    if morsel == INVALID_VALUE {
        return Err(DecodeError::InvalidByte(index_at_start_of_input, input[0]));
    }
    accum = (morsel as u64) << 58;

    let morsel = decode_table[input[1] as usize];
    if morsel == INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 1,
            input[1],
        ));
    }
    accum |= (morsel as u64) << 52;

    let morsel = decode_table[input[2] as usize];
    if morsel == INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 2,
            input[2],
        ));
    }
    accum |= (morsel as u64) << 46;

    let morsel = decode_table[input[3] as usize];
    if morsel == INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 3,
            input[3],
        ));
    }
    accum |= (morsel as u64) << 40;

    let morsel = decode_table[input[4] as usize];
    if morsel == INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 4,
            input[4],
        ));
    }
    accum |= (morsel as u64) << 34;

    let morsel = decode_table[input[5] as usize];
    if morsel == INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 5,
            input[5],
        ));
    }
    accum |= (morsel as u64) << 28;

    let morsel = decode_table[input[6] as usize];
    if morsel == INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 6,
            input[6],
        ));
    }
    accum |= (morsel as u64) << 22;

    let morsel = decode_table[input[7] as usize];
    if morsel == INVALID_VALUE {
        return Err(DecodeError::InvalidByte(
            index_at_start_of_input + 7,
            input[7],
        ));
    }
    accum |= (morsel as u64) << 16;

    write_u64(output, accum);

    Ok(())
}

#[inline]
fn write_u64(output: &mut [u8], value: u64) {
    output[..8].copy_from_slice(&value.to_be_bytes());
}

/// Decode an 8-byte chunk, but only write the 6 bytes actually decoded instead of including 2
/// trailing garbage bytes.
#[inline]
fn decode_chunk_precise(
    input: &[u8],
    index_at_start_of_input: usize,
    decode_table: &[u8; 256],
    output: &mut [u8],
) -> Result<(), DecodeError> {
    let mut tmp_buf = [0_u8; 8];

    decode_chunk(
        input,
        index_at_start_of_input,
        decode_table,
        &mut tmp_buf[..],
    )?;

    output[0..6].copy_from_slice(&tmp_buf[0..6]);

    Ok(())
}

/// Return the number of input chunks (including a possibly partial final chunk) in the input
fn num_chunks(input: &[u8]) -> usize {
    input
        .len()
        .checked_add(INPUT_CHUNK_LEN - 1)
        .expect("Overflow when calculating number of chunks in input")
        / INPUT_CHUNK_LEN
}

/// Errors that can occur while decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// An invalid byte was found in the input. The offset and offending byte are provided.
    InvalidByte(usize, u8),
    /// The length of the input is invalid.
    /// A typical cause of this is stray trailing whitespace or other separator bytes.
    InvalidLength,
    /// The last non-padding input symbol's encoded 6 bits have nonzero bits that will be discarded.
    /// This is indicative of corrupted or truncated Base64.
    /// Unlike InvalidByte, which reports symbols that aren't in the alphabet, this error is for
    /// symbols that are in the alphabet but represent nonsensical encodings.
    InvalidLastSymbol(usize, u8),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            DecodeError::InvalidByte(index, byte) => {
                write!(f, "Invalid byte {}, offset {}.", byte, index)
            }
            DecodeError::InvalidLength => write!(
                f,
                "Encoded text cannot have a 6-bit remainder. Trailing whitespace or other bytes?"
            ),
            DecodeError::InvalidLastSymbol(index, byte) => {
                write!(f, "Invalid last symbol {}, offset {}.", byte, index)
            }
        }
    }
}

#[cfg(any(feature = "std", test))]
impl error::Error for DecodeError {
    fn description(&self) -> &str {
        match *self {
            DecodeError::InvalidByte(_, _) => "invalid byte",
            DecodeError::InvalidLength => "invalid length",
            DecodeError::InvalidLastSymbol(_, _) => "invalid last symbol",
        }
    }

    fn cause(&self) -> Option<&dyn error::Error> {
        None
    }
}

/// B64-encode and pad (if configured).
///
/// This helper exists to avoid recalculating encoded_size, which is relatively expensive on short
/// inputs.
///
/// `encoded_size` is the encoded size calculated for `input`.
///
/// `output` must be of size `encoded_size`.
///
/// All bytes in `output` will be written to since it is exactly the size of the output.
fn encode_with_padding(input: &[u8], config: Config, encoded_size: usize, output: &mut [u8]) {
    debug_assert_eq!(encoded_size, output.len());

    let b64_bytes_written = encode_to_slice(input, output, config.char_set.encode_table());

    let padding_bytes = if config.pad {
        add_padding(input.len(), &mut output[b64_bytes_written..])
    } else {
        0
    };

    let encoded_bytes = b64_bytes_written
        .checked_add(padding_bytes)
        .expect("usize overflow when calculating b64 length");

    debug_assert_eq!(encoded_size, encoded_bytes);
}

/// Write padding characters.
/// `output` is the slice where padding should be written, of length at least 2.
///
/// Returns the number of padding bytes written.
fn add_padding(input_len: usize, output: &mut [u8]) -> usize {
    let rem = input_len % 3;
    let mut bytes_written = 0;
    for _ in 0..((3 - rem) % 3) {
        output[bytes_written] = b'=';
        bytes_written += 1;
    }

    bytes_written
}

/// Encode input bytes to utf8 base64 bytes. Does not pad.
/// `output` must be long enough to hold the encoded `input` without padding.
/// Returns the number of bytes written.
#[inline]
fn encode_to_slice(input: &[u8], output: &mut [u8], encode_table: &[u8; 64]) -> usize {
    let mut input_index: usize = 0;

    const BLOCKS_PER_FAST_LOOP: usize = 4;
    const LOW_SIX_BITS: u64 = 0x3F;

    // we read 8 bytes at a time (u64) but only actually consume 6 of those bytes. Thus, we need
    // 2 trailing bytes to be available to read..
    let last_fast_index = input.len().saturating_sub(BLOCKS_PER_FAST_LOOP * 6 + 2);
    let mut output_index = 0;

    if last_fast_index > 0 {
        while input_index <= last_fast_index {
            // Major performance wins from letting the optimizer do the bounds check once, mostly
            // on the output side
            let input_chunk = &input[input_index..(input_index + (BLOCKS_PER_FAST_LOOP * 6 + 2))];
            let output_chunk = &mut output[output_index..(output_index + BLOCKS_PER_FAST_LOOP * 8)];

            // Hand-unrolling for 32 vs 16 or 8 bytes produces yields performance about equivalent
            // to unsafe pointer code on a Xeon E5-1650v3. 64 byte unrolling was slightly better for
            // large inputs but significantly worse for 50-byte input, unsurprisingly. I suspect
            // that it's a not uncommon use case to encode smallish chunks of data (e.g. a 64-byte
            // SHA-512 digest), so it would be nice if that fit in the unrolled loop at least once.
            // Plus, single-digit percentage performance differences might well be quite different
            // on different hardware.

            let input_u64 = read_u64(&input_chunk[0..]);

            output_chunk[0] = encode_table[((input_u64 >> 58) & LOW_SIX_BITS) as usize];
            output_chunk[1] = encode_table[((input_u64 >> 52) & LOW_SIX_BITS) as usize];
            output_chunk[2] = encode_table[((input_u64 >> 46) & LOW_SIX_BITS) as usize];
            output_chunk[3] = encode_table[((input_u64 >> 40) & LOW_SIX_BITS) as usize];
            output_chunk[4] = encode_table[((input_u64 >> 34) & LOW_SIX_BITS) as usize];
            output_chunk[5] = encode_table[((input_u64 >> 28) & LOW_SIX_BITS) as usize];
            output_chunk[6] = encode_table[((input_u64 >> 22) & LOW_SIX_BITS) as usize];
            output_chunk[7] = encode_table[((input_u64 >> 16) & LOW_SIX_BITS) as usize];

            let input_u64 = read_u64(&input_chunk[6..]);

            output_chunk[8] = encode_table[((input_u64 >> 58) & LOW_SIX_BITS) as usize];
            output_chunk[9] = encode_table[((input_u64 >> 52) & LOW_SIX_BITS) as usize];
            output_chunk[10] = encode_table[((input_u64 >> 46) & LOW_SIX_BITS) as usize];
            output_chunk[11] = encode_table[((input_u64 >> 40) & LOW_SIX_BITS) as usize];
            output_chunk[12] = encode_table[((input_u64 >> 34) & LOW_SIX_BITS) as usize];
            output_chunk[13] = encode_table[((input_u64 >> 28) & LOW_SIX_BITS) as usize];
            output_chunk[14] = encode_table[((input_u64 >> 22) & LOW_SIX_BITS) as usize];
            output_chunk[15] = encode_table[((input_u64 >> 16) & LOW_SIX_BITS) as usize];

            let input_u64 = read_u64(&input_chunk[12..]);

            output_chunk[16] = encode_table[((input_u64 >> 58) & LOW_SIX_BITS) as usize];
            output_chunk[17] = encode_table[((input_u64 >> 52) & LOW_SIX_BITS) as usize];
            output_chunk[18] = encode_table[((input_u64 >> 46) & LOW_SIX_BITS) as usize];
            output_chunk[19] = encode_table[((input_u64 >> 40) & LOW_SIX_BITS) as usize];
            output_chunk[20] = encode_table[((input_u64 >> 34) & LOW_SIX_BITS) as usize];
            output_chunk[21] = encode_table[((input_u64 >> 28) & LOW_SIX_BITS) as usize];
            output_chunk[22] = encode_table[((input_u64 >> 22) & LOW_SIX_BITS) as usize];
            output_chunk[23] = encode_table[((input_u64 >> 16) & LOW_SIX_BITS) as usize];

            let input_u64 = read_u64(&input_chunk[18..]);

            output_chunk[24] = encode_table[((input_u64 >> 58) & LOW_SIX_BITS) as usize];
            output_chunk[25] = encode_table[((input_u64 >> 52) & LOW_SIX_BITS) as usize];
            output_chunk[26] = encode_table[((input_u64 >> 46) & LOW_SIX_BITS) as usize];
            output_chunk[27] = encode_table[((input_u64 >> 40) & LOW_SIX_BITS) as usize];
            output_chunk[28] = encode_table[((input_u64 >> 34) & LOW_SIX_BITS) as usize];
            output_chunk[29] = encode_table[((input_u64 >> 28) & LOW_SIX_BITS) as usize];
            output_chunk[30] = encode_table[((input_u64 >> 22) & LOW_SIX_BITS) as usize];
            output_chunk[31] = encode_table[((input_u64 >> 16) & LOW_SIX_BITS) as usize];

            output_index += BLOCKS_PER_FAST_LOOP * 8;
            input_index += BLOCKS_PER_FAST_LOOP * 6;
        }
    }

    // Encode what's left after the fast loop.

    const LOW_SIX_BITS_U8: u8 = 0x3F;

    let rem = input.len() % 3;
    let start_of_rem = input.len() - rem;

    // start at the first index not handled by fast loop, which may be 0.

    while input_index < start_of_rem {
        let input_chunk = &input[input_index..(input_index + 3)];
        let output_chunk = &mut output[output_index..(output_index + 4)];

        output_chunk[0] = encode_table[(input_chunk[0] >> 2) as usize];
        output_chunk[1] =
            encode_table[((input_chunk[0] << 4 | input_chunk[1] >> 4) & LOW_SIX_BITS_U8) as usize];
        output_chunk[2] =
            encode_table[((input_chunk[1] << 2 | input_chunk[2] >> 6) & LOW_SIX_BITS_U8) as usize];
        output_chunk[3] = encode_table[(input_chunk[2] & LOW_SIX_BITS_U8) as usize];

        input_index += 3;
        output_index += 4;
    }

    if rem == 2 {
        output[output_index] = encode_table[(input[start_of_rem] >> 2) as usize];
        output[output_index + 1] = encode_table[((input[start_of_rem] << 4
            | input[start_of_rem + 1] >> 4)
            & LOW_SIX_BITS_U8) as usize];
        output[output_index + 2] =
            encode_table[((input[start_of_rem + 1] << 2) & LOW_SIX_BITS_U8) as usize];
        output_index += 3;
    } else if rem == 1 {
        output[output_index] = encode_table[(input[start_of_rem] >> 2) as usize];
        output[output_index + 1] =
            encode_table[((input[start_of_rem] << 4) & LOW_SIX_BITS_U8) as usize];
        output_index += 2;
    }

    output_index
}

#[inline]
fn read_u64(s: &[u8]) -> u64 {
    u64::from_be_bytes(s[..8].try_into().unwrap())
}

/// calculate the base64 encoded string size, including padding if appropriate
fn encoded_size(bytes_len: usize, config: Config) -> Option<usize> {
    let rem = bytes_len % 3;

    let complete_input_chunks = bytes_len / 3;
    let complete_chunk_output = complete_input_chunks.checked_mul(4);

    if rem > 0 {
        if config.pad {
            complete_chunk_output.and_then(|c| c.checked_add(4))
        } else {
            let encoded_rem = match rem {
                1 => 2,
                2 => 3,
                _ => unreachable!("Impossible remainder"),
            };
            complete_chunk_output.and_then(|c| c.checked_add(encoded_rem))
        }
    } else {
        complete_chunk_output
    }
}

/// Contains configuration parameters for base64 encoding
#[derive(Clone, Copy, Debug)]
struct Config {
    /// Character set to use
    char_set: CharacterSet,
    /// True to pad output with `=` characters
    pad: bool,
    /// True to ignore excess nonzero bits in the last few symbols, otherwise an error is returned.
    decode_allow_trailing_bits: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum CharacterSet {
    /// The standard character set (uses `+` and `/`).
    ///
    /// See [RFC 3548](https://tools.ietf.org/html/rfc3548#section-3).
    Standard,
    /// The URL safe character set (uses `-` and `_`).
    ///
    /// See [RFC 3548](https://tools.ietf.org/html/rfc3548#section-4).
    UrlSafe,
    /// The `crypt(3)` character set (uses `./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz`).
    ///
    /// Not standardized, but folk wisdom on the net asserts that this alphabet is what crypt uses.
    Crypt,
    /// The bcrypt character set (uses `./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789`).
    Bcrypt,
    /// The character set used in IMAP-modified UTF-7 (uses `+` and `,`).
    ///
    /// See [RFC 3501](https://tools.ietf.org/html/rfc3501#section-5.1.3)
    ImapMutf7,
    /// The character set used in BinHex 4.0 files.
    ///
    /// See [BinHex 4.0 Definition](http://files.stairways.com/other/binhex-40-specs-info.txt)
    BinHex,
}

impl CharacterSet {
    fn encode_table(self) -> &'static [u8; 64] {
        match self {
            CharacterSet::Standard => STANDARD_ENCODE,
            CharacterSet::UrlSafe => URL_SAFE_ENCODE,
            CharacterSet::Crypt => CRYPT_ENCODE,
            CharacterSet::Bcrypt => BCRYPT_ENCODE,
            CharacterSet::ImapMutf7 => IMAP_MUTF7_ENCODE,
            CharacterSet::BinHex => BINHEX_ENCODE,
        }
    }

    fn decode_table(self) -> &'static [u8; 256] {
        match self {
            CharacterSet::Standard => STANDARD_DECODE,
            CharacterSet::UrlSafe => URL_SAFE_DECODE,
            CharacterSet::Crypt => CRYPT_DECODE,
            CharacterSet::Bcrypt => BCRYPT_DECODE,
            CharacterSet::ImapMutf7 => IMAP_MUTF7_DECODE,
            CharacterSet::BinHex => BINHEX_DECODE,
        }
    }
}

/// Standard character set with padding.
const STANDARD: Config = Config {
    char_set: CharacterSet::Standard,
    pad: true,
    decode_allow_trailing_bits: false,
};

// decode logic operates on chunks of 8 input bytes without padding
const INPUT_CHUNK_LEN: usize = 8;
const DECODED_CHUNK_LEN: usize = 6;
// we read a u64 and write a u64, but a u64 of input only yields 6 bytes of output, so the last
// 2 bytes of any output u64 should not be counted as written to (but must be available in a
// slice).
const DECODED_CHUNK_SUFFIX: usize = 2;

// how many u64's of input to handle at a time
const CHUNKS_PER_FAST_LOOP_BLOCK: usize = 4;
const INPUT_BLOCK_LEN: usize = CHUNKS_PER_FAST_LOOP_BLOCK * INPUT_CHUNK_LEN;
// includes the trailing 2 bytes for the final u64 write
const DECODED_BLOCK_LEN: usize =
    CHUNKS_PER_FAST_LOOP_BLOCK * DECODED_CHUNK_LEN + DECODED_CHUNK_SUFFIX;

const INVALID_VALUE: u8 = 255;
#[rustfmt::skip]
 const STANDARD_ENCODE: &[u8; 64] = &[
    65, // input 0 (0x0) => 'A' (0x41)
    66, // input 1 (0x1) => 'B' (0x42)
    67, // input 2 (0x2) => 'C' (0x43)
    68, // input 3 (0x3) => 'D' (0x44)
    69, // input 4 (0x4) => 'E' (0x45)
    70, // input 5 (0x5) => 'F' (0x46)
    71, // input 6 (0x6) => 'G' (0x47)
    72, // input 7 (0x7) => 'H' (0x48)
    73, // input 8 (0x8) => 'I' (0x49)
    74, // input 9 (0x9) => 'J' (0x4A)
    75, // input 10 (0xA) => 'K' (0x4B)
    76, // input 11 (0xB) => 'L' (0x4C)
    77, // input 12 (0xC) => 'M' (0x4D)
    78, // input 13 (0xD) => 'N' (0x4E)
    79, // input 14 (0xE) => 'O' (0x4F)
    80, // input 15 (0xF) => 'P' (0x50)
    81, // input 16 (0x10) => 'Q' (0x51)
    82, // input 17 (0x11) => 'R' (0x52)
    83, // input 18 (0x12) => 'S' (0x53)
    84, // input 19 (0x13) => 'T' (0x54)
    85, // input 20 (0x14) => 'U' (0x55)
    86, // input 21 (0x15) => 'V' (0x56)
    87, // input 22 (0x16) => 'W' (0x57)
    88, // input 23 (0x17) => 'X' (0x58)
    89, // input 24 (0x18) => 'Y' (0x59)
    90, // input 25 (0x19) => 'Z' (0x5A)
    97, // input 26 (0x1A) => 'a' (0x61)
    98, // input 27 (0x1B) => 'b' (0x62)
    99, // input 28 (0x1C) => 'c' (0x63)
    100, // input 29 (0x1D) => 'd' (0x64)
    101, // input 30 (0x1E) => 'e' (0x65)
    102, // input 31 (0x1F) => 'f' (0x66)
    103, // input 32 (0x20) => 'g' (0x67)
    104, // input 33 (0x21) => 'h' (0x68)
    105, // input 34 (0x22) => 'i' (0x69)
    106, // input 35 (0x23) => 'j' (0x6A)
    107, // input 36 (0x24) => 'k' (0x6B)
    108, // input 37 (0x25) => 'l' (0x6C)
    109, // input 38 (0x26) => 'm' (0x6D)
    110, // input 39 (0x27) => 'n' (0x6E)
    111, // input 40 (0x28) => 'o' (0x6F)
    112, // input 41 (0x29) => 'p' (0x70)
    113, // input 42 (0x2A) => 'q' (0x71)
    114, // input 43 (0x2B) => 'r' (0x72)
    115, // input 44 (0x2C) => 's' (0x73)
    116, // input 45 (0x2D) => 't' (0x74)
    117, // input 46 (0x2E) => 'u' (0x75)
    118, // input 47 (0x2F) => 'v' (0x76)
    119, // input 48 (0x30) => 'w' (0x77)
    120, // input 49 (0x31) => 'x' (0x78)
    121, // input 50 (0x32) => 'y' (0x79)
    122, // input 51 (0x33) => 'z' (0x7A)
    48, // input 52 (0x34) => '0' (0x30)
    49, // input 53 (0x35) => '1' (0x31)
    50, // input 54 (0x36) => '2' (0x32)
    51, // input 55 (0x37) => '3' (0x33)
    52, // input 56 (0x38) => '4' (0x34)
    53, // input 57 (0x39) => '5' (0x35)
    54, // input 58 (0x3A) => '6' (0x36)
    55, // input 59 (0x3B) => '7' (0x37)
    56, // input 60 (0x3C) => '8' (0x38)
    57, // input 61 (0x3D) => '9' (0x39)
    43, // input 62 (0x3E) => '+' (0x2B)
    47, // input 63 (0x3F) => '/' (0x2F)
];
#[rustfmt::skip]
 const STANDARD_DECODE: &[u8; 256] = &[
    INVALID_VALUE, // input 0 (0x0)
    INVALID_VALUE, // input 1 (0x1)
    INVALID_VALUE, // input 2 (0x2)
    INVALID_VALUE, // input 3 (0x3)
    INVALID_VALUE, // input 4 (0x4)
    INVALID_VALUE, // input 5 (0x5)
    INVALID_VALUE, // input 6 (0x6)
    INVALID_VALUE, // input 7 (0x7)
    INVALID_VALUE, // input 8 (0x8)
    INVALID_VALUE, // input 9 (0x9)
    INVALID_VALUE, // input 10 (0xA)
    INVALID_VALUE, // input 11 (0xB)
    INVALID_VALUE, // input 12 (0xC)
    INVALID_VALUE, // input 13 (0xD)
    INVALID_VALUE, // input 14 (0xE)
    INVALID_VALUE, // input 15 (0xF)
    INVALID_VALUE, // input 16 (0x10)
    INVALID_VALUE, // input 17 (0x11)
    INVALID_VALUE, // input 18 (0x12)
    INVALID_VALUE, // input 19 (0x13)
    INVALID_VALUE, // input 20 (0x14)
    INVALID_VALUE, // input 21 (0x15)
    INVALID_VALUE, // input 22 (0x16)
    INVALID_VALUE, // input 23 (0x17)
    INVALID_VALUE, // input 24 (0x18)
    INVALID_VALUE, // input 25 (0x19)
    INVALID_VALUE, // input 26 (0x1A)
    INVALID_VALUE, // input 27 (0x1B)
    INVALID_VALUE, // input 28 (0x1C)
    INVALID_VALUE, // input 29 (0x1D)
    INVALID_VALUE, // input 30 (0x1E)
    INVALID_VALUE, // input 31 (0x1F)
    INVALID_VALUE, // input 32 (0x20)
    INVALID_VALUE, // input 33 (0x21)
    INVALID_VALUE, // input 34 (0x22)
    INVALID_VALUE, // input 35 (0x23)
    INVALID_VALUE, // input 36 (0x24)
    INVALID_VALUE, // input 37 (0x25)
    INVALID_VALUE, // input 38 (0x26)
    INVALID_VALUE, // input 39 (0x27)
    INVALID_VALUE, // input 40 (0x28)
    INVALID_VALUE, // input 41 (0x29)
    INVALID_VALUE, // input 42 (0x2A)
    62, // input 43 (0x2B char '+') => 62 (0x3E)
    INVALID_VALUE, // input 44 (0x2C)
    INVALID_VALUE, // input 45 (0x2D)
    INVALID_VALUE, // input 46 (0x2E)
    63, // input 47 (0x2F char '/') => 63 (0x3F)
    52, // input 48 (0x30 char '0') => 52 (0x34)
    53, // input 49 (0x31 char '1') => 53 (0x35)
    54, // input 50 (0x32 char '2') => 54 (0x36)
    55, // input 51 (0x33 char '3') => 55 (0x37)
    56, // input 52 (0x34 char '4') => 56 (0x38)
    57, // input 53 (0x35 char '5') => 57 (0x39)
    58, // input 54 (0x36 char '6') => 58 (0x3A)
    59, // input 55 (0x37 char '7') => 59 (0x3B)
    60, // input 56 (0x38 char '8') => 60 (0x3C)
    61, // input 57 (0x39 char '9') => 61 (0x3D)
    INVALID_VALUE, // input 58 (0x3A)
    INVALID_VALUE, // input 59 (0x3B)
    INVALID_VALUE, // input 60 (0x3C)
    INVALID_VALUE, // input 61 (0x3D)
    INVALID_VALUE, // input 62 (0x3E)
    INVALID_VALUE, // input 63 (0x3F)
    INVALID_VALUE, // input 64 (0x40)
    0, // input 65 (0x41 char 'A') => 0 (0x0)
    1, // input 66 (0x42 char 'B') => 1 (0x1)
    2, // input 67 (0x43 char 'C') => 2 (0x2)
    3, // input 68 (0x44 char 'D') => 3 (0x3)
    4, // input 69 (0x45 char 'E') => 4 (0x4)
    5, // input 70 (0x46 char 'F') => 5 (0x5)
    6, // input 71 (0x47 char 'G') => 6 (0x6)
    7, // input 72 (0x48 char 'H') => 7 (0x7)
    8, // input 73 (0x49 char 'I') => 8 (0x8)
    9, // input 74 (0x4A char 'J') => 9 (0x9)
    10, // input 75 (0x4B char 'K') => 10 (0xA)
    11, // input 76 (0x4C char 'L') => 11 (0xB)
    12, // input 77 (0x4D char 'M') => 12 (0xC)
    13, // input 78 (0x4E char 'N') => 13 (0xD)
    14, // input 79 (0x4F char 'O') => 14 (0xE)
    15, // input 80 (0x50 char 'P') => 15 (0xF)
    16, // input 81 (0x51 char 'Q') => 16 (0x10)
    17, // input 82 (0x52 char 'R') => 17 (0x11)
    18, // input 83 (0x53 char 'S') => 18 (0x12)
    19, // input 84 (0x54 char 'T') => 19 (0x13)
    20, // input 85 (0x55 char 'U') => 20 (0x14)
    21, // input 86 (0x56 char 'V') => 21 (0x15)
    22, // input 87 (0x57 char 'W') => 22 (0x16)
    23, // input 88 (0x58 char 'X') => 23 (0x17)
    24, // input 89 (0x59 char 'Y') => 24 (0x18)
    25, // input 90 (0x5A char 'Z') => 25 (0x19)
    INVALID_VALUE, // input 91 (0x5B)
    INVALID_VALUE, // input 92 (0x5C)
    INVALID_VALUE, // input 93 (0x5D)
    INVALID_VALUE, // input 94 (0x5E)
    INVALID_VALUE, // input 95 (0x5F)
    INVALID_VALUE, // input 96 (0x60)
    26, // input 97 (0x61 char 'a') => 26 (0x1A)
    27, // input 98 (0x62 char 'b') => 27 (0x1B)
    28, // input 99 (0x63 char 'c') => 28 (0x1C)
    29, // input 100 (0x64 char 'd') => 29 (0x1D)
    30, // input 101 (0x65 char 'e') => 30 (0x1E)
    31, // input 102 (0x66 char 'f') => 31 (0x1F)
    32, // input 103 (0x67 char 'g') => 32 (0x20)
    33, // input 104 (0x68 char 'h') => 33 (0x21)
    34, // input 105 (0x69 char 'i') => 34 (0x22)
    35, // input 106 (0x6A char 'j') => 35 (0x23)
    36, // input 107 (0x6B char 'k') => 36 (0x24)
    37, // input 108 (0x6C char 'l') => 37 (0x25)
    38, // input 109 (0x6D char 'm') => 38 (0x26)
    39, // input 110 (0x6E char 'n') => 39 (0x27)
    40, // input 111 (0x6F char 'o') => 40 (0x28)
    41, // input 112 (0x70 char 'p') => 41 (0x29)
    42, // input 113 (0x71 char 'q') => 42 (0x2A)
    43, // input 114 (0x72 char 'r') => 43 (0x2B)
    44, // input 115 (0x73 char 's') => 44 (0x2C)
    45, // input 116 (0x74 char 't') => 45 (0x2D)
    46, // input 117 (0x75 char 'u') => 46 (0x2E)
    47, // input 118 (0x76 char 'v') => 47 (0x2F)
    48, // input 119 (0x77 char 'w') => 48 (0x30)
    49, // input 120 (0x78 char 'x') => 49 (0x31)
    50, // input 121 (0x79 char 'y') => 50 (0x32)
    51, // input 122 (0x7A char 'z') => 51 (0x33)
    INVALID_VALUE, // input 123 (0x7B)
    INVALID_VALUE, // input 124 (0x7C)
    INVALID_VALUE, // input 125 (0x7D)
    INVALID_VALUE, // input 126 (0x7E)
    INVALID_VALUE, // input 127 (0x7F)
    INVALID_VALUE, // input 128 (0x80)
    INVALID_VALUE, // input 129 (0x81)
    INVALID_VALUE, // input 130 (0x82)
    INVALID_VALUE, // input 131 (0x83)
    INVALID_VALUE, // input 132 (0x84)
    INVALID_VALUE, // input 133 (0x85)
    INVALID_VALUE, // input 134 (0x86)
    INVALID_VALUE, // input 135 (0x87)
    INVALID_VALUE, // input 136 (0x88)
    INVALID_VALUE, // input 137 (0x89)
    INVALID_VALUE, // input 138 (0x8A)
    INVALID_VALUE, // input 139 (0x8B)
    INVALID_VALUE, // input 140 (0x8C)
    INVALID_VALUE, // input 141 (0x8D)
    INVALID_VALUE, // input 142 (0x8E)
    INVALID_VALUE, // input 143 (0x8F)
    INVALID_VALUE, // input 144 (0x90)
    INVALID_VALUE, // input 145 (0x91)
    INVALID_VALUE, // input 146 (0x92)
    INVALID_VALUE, // input 147 (0x93)
    INVALID_VALUE, // input 148 (0x94)
    INVALID_VALUE, // input 149 (0x95)
    INVALID_VALUE, // input 150 (0x96)
    INVALID_VALUE, // input 151 (0x97)
    INVALID_VALUE, // input 152 (0x98)
    INVALID_VALUE, // input 153 (0x99)
    INVALID_VALUE, // input 154 (0x9A)
    INVALID_VALUE, // input 155 (0x9B)
    INVALID_VALUE, // input 156 (0x9C)
    INVALID_VALUE, // input 157 (0x9D)
    INVALID_VALUE, // input 158 (0x9E)
    INVALID_VALUE, // input 159 (0x9F)
    INVALID_VALUE, // input 160 (0xA0)
    INVALID_VALUE, // input 161 (0xA1)
    INVALID_VALUE, // input 162 (0xA2)
    INVALID_VALUE, // input 163 (0xA3)
    INVALID_VALUE, // input 164 (0xA4)
    INVALID_VALUE, // input 165 (0xA5)
    INVALID_VALUE, // input 166 (0xA6)
    INVALID_VALUE, // input 167 (0xA7)
    INVALID_VALUE, // input 168 (0xA8)
    INVALID_VALUE, // input 169 (0xA9)
    INVALID_VALUE, // input 170 (0xAA)
    INVALID_VALUE, // input 171 (0xAB)
    INVALID_VALUE, // input 172 (0xAC)
    INVALID_VALUE, // input 173 (0xAD)
    INVALID_VALUE, // input 174 (0xAE)
    INVALID_VALUE, // input 175 (0xAF)
    INVALID_VALUE, // input 176 (0xB0)
    INVALID_VALUE, // input 177 (0xB1)
    INVALID_VALUE, // input 178 (0xB2)
    INVALID_VALUE, // input 179 (0xB3)
    INVALID_VALUE, // input 180 (0xB4)
    INVALID_VALUE, // input 181 (0xB5)
    INVALID_VALUE, // input 182 (0xB6)
    INVALID_VALUE, // input 183 (0xB7)
    INVALID_VALUE, // input 184 (0xB8)
    INVALID_VALUE, // input 185 (0xB9)
    INVALID_VALUE, // input 186 (0xBA)
    INVALID_VALUE, // input 187 (0xBB)
    INVALID_VALUE, // input 188 (0xBC)
    INVALID_VALUE, // input 189 (0xBD)
    INVALID_VALUE, // input 190 (0xBE)
    INVALID_VALUE, // input 191 (0xBF)
    INVALID_VALUE, // input 192 (0xC0)
    INVALID_VALUE, // input 193 (0xC1)
    INVALID_VALUE, // input 194 (0xC2)
    INVALID_VALUE, // input 195 (0xC3)
    INVALID_VALUE, // input 196 (0xC4)
    INVALID_VALUE, // input 197 (0xC5)
    INVALID_VALUE, // input 198 (0xC6)
    INVALID_VALUE, // input 199 (0xC7)
    INVALID_VALUE, // input 200 (0xC8)
    INVALID_VALUE, // input 201 (0xC9)
    INVALID_VALUE, // input 202 (0xCA)
    INVALID_VALUE, // input 203 (0xCB)
    INVALID_VALUE, // input 204 (0xCC)
    INVALID_VALUE, // input 205 (0xCD)
    INVALID_VALUE, // input 206 (0xCE)
    INVALID_VALUE, // input 207 (0xCF)
    INVALID_VALUE, // input 208 (0xD0)
    INVALID_VALUE, // input 209 (0xD1)
    INVALID_VALUE, // input 210 (0xD2)
    INVALID_VALUE, // input 211 (0xD3)
    INVALID_VALUE, // input 212 (0xD4)
    INVALID_VALUE, // input 213 (0xD5)
    INVALID_VALUE, // input 214 (0xD6)
    INVALID_VALUE, // input 215 (0xD7)
    INVALID_VALUE, // input 216 (0xD8)
    INVALID_VALUE, // input 217 (0xD9)
    INVALID_VALUE, // input 218 (0xDA)
    INVALID_VALUE, // input 219 (0xDB)
    INVALID_VALUE, // input 220 (0xDC)
    INVALID_VALUE, // input 221 (0xDD)
    INVALID_VALUE, // input 222 (0xDE)
    INVALID_VALUE, // input 223 (0xDF)
    INVALID_VALUE, // input 224 (0xE0)
    INVALID_VALUE, // input 225 (0xE1)
    INVALID_VALUE, // input 226 (0xE2)
    INVALID_VALUE, // input 227 (0xE3)
    INVALID_VALUE, // input 228 (0xE4)
    INVALID_VALUE, // input 229 (0xE5)
    INVALID_VALUE, // input 230 (0xE6)
    INVALID_VALUE, // input 231 (0xE7)
    INVALID_VALUE, // input 232 (0xE8)
    INVALID_VALUE, // input 233 (0xE9)
    INVALID_VALUE, // input 234 (0xEA)
    INVALID_VALUE, // input 235 (0xEB)
    INVALID_VALUE, // input 236 (0xEC)
    INVALID_VALUE, // input 237 (0xED)
    INVALID_VALUE, // input 238 (0xEE)
    INVALID_VALUE, // input 239 (0xEF)
    INVALID_VALUE, // input 240 (0xF0)
    INVALID_VALUE, // input 241 (0xF1)
    INVALID_VALUE, // input 242 (0xF2)
    INVALID_VALUE, // input 243 (0xF3)
    INVALID_VALUE, // input 244 (0xF4)
    INVALID_VALUE, // input 245 (0xF5)
    INVALID_VALUE, // input 246 (0xF6)
    INVALID_VALUE, // input 247 (0xF7)
    INVALID_VALUE, // input 248 (0xF8)
    INVALID_VALUE, // input 249 (0xF9)
    INVALID_VALUE, // input 250 (0xFA)
    INVALID_VALUE, // input 251 (0xFB)
    INVALID_VALUE, // input 252 (0xFC)
    INVALID_VALUE, // input 253 (0xFD)
    INVALID_VALUE, // input 254 (0xFE)
    INVALID_VALUE, // input 255 (0xFF)
];
#[rustfmt::skip]
 const URL_SAFE_ENCODE: &[u8; 64] = &[
    65, // input 0 (0x0) => 'A' (0x41)
    66, // input 1 (0x1) => 'B' (0x42)
    67, // input 2 (0x2) => 'C' (0x43)
    68, // input 3 (0x3) => 'D' (0x44)
    69, // input 4 (0x4) => 'E' (0x45)
    70, // input 5 (0x5) => 'F' (0x46)
    71, // input 6 (0x6) => 'G' (0x47)
    72, // input 7 (0x7) => 'H' (0x48)
    73, // input 8 (0x8) => 'I' (0x49)
    74, // input 9 (0x9) => 'J' (0x4A)
    75, // input 10 (0xA) => 'K' (0x4B)
    76, // input 11 (0xB) => 'L' (0x4C)
    77, // input 12 (0xC) => 'M' (0x4D)
    78, // input 13 (0xD) => 'N' (0x4E)
    79, // input 14 (0xE) => 'O' (0x4F)
    80, // input 15 (0xF) => 'P' (0x50)
    81, // input 16 (0x10) => 'Q' (0x51)
    82, // input 17 (0x11) => 'R' (0x52)
    83, // input 18 (0x12) => 'S' (0x53)
    84, // input 19 (0x13) => 'T' (0x54)
    85, // input 20 (0x14) => 'U' (0x55)
    86, // input 21 (0x15) => 'V' (0x56)
    87, // input 22 (0x16) => 'W' (0x57)
    88, // input 23 (0x17) => 'X' (0x58)
    89, // input 24 (0x18) => 'Y' (0x59)
    90, // input 25 (0x19) => 'Z' (0x5A)
    97, // input 26 (0x1A) => 'a' (0x61)
    98, // input 27 (0x1B) => 'b' (0x62)
    99, // input 28 (0x1C) => 'c' (0x63)
    100, // input 29 (0x1D) => 'd' (0x64)
    101, // input 30 (0x1E) => 'e' (0x65)
    102, // input 31 (0x1F) => 'f' (0x66)
    103, // input 32 (0x20) => 'g' (0x67)
    104, // input 33 (0x21) => 'h' (0x68)
    105, // input 34 (0x22) => 'i' (0x69)
    106, // input 35 (0x23) => 'j' (0x6A)
    107, // input 36 (0x24) => 'k' (0x6B)
    108, // input 37 (0x25) => 'l' (0x6C)
    109, // input 38 (0x26) => 'm' (0x6D)
    110, // input 39 (0x27) => 'n' (0x6E)
    111, // input 40 (0x28) => 'o' (0x6F)
    112, // input 41 (0x29) => 'p' (0x70)
    113, // input 42 (0x2A) => 'q' (0x71)
    114, // input 43 (0x2B) => 'r' (0x72)
    115, // input 44 (0x2C) => 's' (0x73)
    116, // input 45 (0x2D) => 't' (0x74)
    117, // input 46 (0x2E) => 'u' (0x75)
    118, // input 47 (0x2F) => 'v' (0x76)
    119, // input 48 (0x30) => 'w' (0x77)
    120, // input 49 (0x31) => 'x' (0x78)
    121, // input 50 (0x32) => 'y' (0x79)
    122, // input 51 (0x33) => 'z' (0x7A)
    48, // input 52 (0x34) => '0' (0x30)
    49, // input 53 (0x35) => '1' (0x31)
    50, // input 54 (0x36) => '2' (0x32)
    51, // input 55 (0x37) => '3' (0x33)
    52, // input 56 (0x38) => '4' (0x34)
    53, // input 57 (0x39) => '5' (0x35)
    54, // input 58 (0x3A) => '6' (0x36)
    55, // input 59 (0x3B) => '7' (0x37)
    56, // input 60 (0x3C) => '8' (0x38)
    57, // input 61 (0x3D) => '9' (0x39)
    45, // input 62 (0x3E) => '-' (0x2D)
    95, // input 63 (0x3F) => '_' (0x5F)
];
#[rustfmt::skip]
 const URL_SAFE_DECODE: &[u8; 256] = &[
    INVALID_VALUE, // input 0 (0x0)
    INVALID_VALUE, // input 1 (0x1)
    INVALID_VALUE, // input 2 (0x2)
    INVALID_VALUE, // input 3 (0x3)
    INVALID_VALUE, // input 4 (0x4)
    INVALID_VALUE, // input 5 (0x5)
    INVALID_VALUE, // input 6 (0x6)
    INVALID_VALUE, // input 7 (0x7)
    INVALID_VALUE, // input 8 (0x8)
    INVALID_VALUE, // input 9 (0x9)
    INVALID_VALUE, // input 10 (0xA)
    INVALID_VALUE, // input 11 (0xB)
    INVALID_VALUE, // input 12 (0xC)
    INVALID_VALUE, // input 13 (0xD)
    INVALID_VALUE, // input 14 (0xE)
    INVALID_VALUE, // input 15 (0xF)
    INVALID_VALUE, // input 16 (0x10)
    INVALID_VALUE, // input 17 (0x11)
    INVALID_VALUE, // input 18 (0x12)
    INVALID_VALUE, // input 19 (0x13)
    INVALID_VALUE, // input 20 (0x14)
    INVALID_VALUE, // input 21 (0x15)
    INVALID_VALUE, // input 22 (0x16)
    INVALID_VALUE, // input 23 (0x17)
    INVALID_VALUE, // input 24 (0x18)
    INVALID_VALUE, // input 25 (0x19)
    INVALID_VALUE, // input 26 (0x1A)
    INVALID_VALUE, // input 27 (0x1B)
    INVALID_VALUE, // input 28 (0x1C)
    INVALID_VALUE, // input 29 (0x1D)
    INVALID_VALUE, // input 30 (0x1E)
    INVALID_VALUE, // input 31 (0x1F)
    INVALID_VALUE, // input 32 (0x20)
    INVALID_VALUE, // input 33 (0x21)
    INVALID_VALUE, // input 34 (0x22)
    INVALID_VALUE, // input 35 (0x23)
    INVALID_VALUE, // input 36 (0x24)
    INVALID_VALUE, // input 37 (0x25)
    INVALID_VALUE, // input 38 (0x26)
    INVALID_VALUE, // input 39 (0x27)
    INVALID_VALUE, // input 40 (0x28)
    INVALID_VALUE, // input 41 (0x29)
    INVALID_VALUE, // input 42 (0x2A)
    INVALID_VALUE, // input 43 (0x2B)
    INVALID_VALUE, // input 44 (0x2C)
    62, // input 45 (0x2D char '-') => 62 (0x3E)
    INVALID_VALUE, // input 46 (0x2E)
    INVALID_VALUE, // input 47 (0x2F)
    52, // input 48 (0x30 char '0') => 52 (0x34)
    53, // input 49 (0x31 char '1') => 53 (0x35)
    54, // input 50 (0x32 char '2') => 54 (0x36)
    55, // input 51 (0x33 char '3') => 55 (0x37)
    56, // input 52 (0x34 char '4') => 56 (0x38)
    57, // input 53 (0x35 char '5') => 57 (0x39)
    58, // input 54 (0x36 char '6') => 58 (0x3A)
    59, // input 55 (0x37 char '7') => 59 (0x3B)
    60, // input 56 (0x38 char '8') => 60 (0x3C)
    61, // input 57 (0x39 char '9') => 61 (0x3D)
    INVALID_VALUE, // input 58 (0x3A)
    INVALID_VALUE, // input 59 (0x3B)
    INVALID_VALUE, // input 60 (0x3C)
    INVALID_VALUE, // input 61 (0x3D)
    INVALID_VALUE, // input 62 (0x3E)
    INVALID_VALUE, // input 63 (0x3F)
    INVALID_VALUE, // input 64 (0x40)
    0, // input 65 (0x41 char 'A') => 0 (0x0)
    1, // input 66 (0x42 char 'B') => 1 (0x1)
    2, // input 67 (0x43 char 'C') => 2 (0x2)
    3, // input 68 (0x44 char 'D') => 3 (0x3)
    4, // input 69 (0x45 char 'E') => 4 (0x4)
    5, // input 70 (0x46 char 'F') => 5 (0x5)
    6, // input 71 (0x47 char 'G') => 6 (0x6)
    7, // input 72 (0x48 char 'H') => 7 (0x7)
    8, // input 73 (0x49 char 'I') => 8 (0x8)
    9, // input 74 (0x4A char 'J') => 9 (0x9)
    10, // input 75 (0x4B char 'K') => 10 (0xA)
    11, // input 76 (0x4C char 'L') => 11 (0xB)
    12, // input 77 (0x4D char 'M') => 12 (0xC)
    13, // input 78 (0x4E char 'N') => 13 (0xD)
    14, // input 79 (0x4F char 'O') => 14 (0xE)
    15, // input 80 (0x50 char 'P') => 15 (0xF)
    16, // input 81 (0x51 char 'Q') => 16 (0x10)
    17, // input 82 (0x52 char 'R') => 17 (0x11)
    18, // input 83 (0x53 char 'S') => 18 (0x12)
    19, // input 84 (0x54 char 'T') => 19 (0x13)
    20, // input 85 (0x55 char 'U') => 20 (0x14)
    21, // input 86 (0x56 char 'V') => 21 (0x15)
    22, // input 87 (0x57 char 'W') => 22 (0x16)
    23, // input 88 (0x58 char 'X') => 23 (0x17)
    24, // input 89 (0x59 char 'Y') => 24 (0x18)
    25, // input 90 (0x5A char 'Z') => 25 (0x19)
    INVALID_VALUE, // input 91 (0x5B)
    INVALID_VALUE, // input 92 (0x5C)
    INVALID_VALUE, // input 93 (0x5D)
    INVALID_VALUE, // input 94 (0x5E)
    63, // input 95 (0x5F char '_') => 63 (0x3F)
    INVALID_VALUE, // input 96 (0x60)
    26, // input 97 (0x61 char 'a') => 26 (0x1A)
    27, // input 98 (0x62 char 'b') => 27 (0x1B)
    28, // input 99 (0x63 char 'c') => 28 (0x1C)
    29, // input 100 (0x64 char 'd') => 29 (0x1D)
    30, // input 101 (0x65 char 'e') => 30 (0x1E)
    31, // input 102 (0x66 char 'f') => 31 (0x1F)
    32, // input 103 (0x67 char 'g') => 32 (0x20)
    33, // input 104 (0x68 char 'h') => 33 (0x21)
    34, // input 105 (0x69 char 'i') => 34 (0x22)
    35, // input 106 (0x6A char 'j') => 35 (0x23)
    36, // input 107 (0x6B char 'k') => 36 (0x24)
    37, // input 108 (0x6C char 'l') => 37 (0x25)
    38, // input 109 (0x6D char 'm') => 38 (0x26)
    39, // input 110 (0x6E char 'n') => 39 (0x27)
    40, // input 111 (0x6F char 'o') => 40 (0x28)
    41, // input 112 (0x70 char 'p') => 41 (0x29)
    42, // input 113 (0x71 char 'q') => 42 (0x2A)
    43, // input 114 (0x72 char 'r') => 43 (0x2B)
    44, // input 115 (0x73 char 's') => 44 (0x2C)
    45, // input 116 (0x74 char 't') => 45 (0x2D)
    46, // input 117 (0x75 char 'u') => 46 (0x2E)
    47, // input 118 (0x76 char 'v') => 47 (0x2F)
    48, // input 119 (0x77 char 'w') => 48 (0x30)
    49, // input 120 (0x78 char 'x') => 49 (0x31)
    50, // input 121 (0x79 char 'y') => 50 (0x32)
    51, // input 122 (0x7A char 'z') => 51 (0x33)
    INVALID_VALUE, // input 123 (0x7B)
    INVALID_VALUE, // input 124 (0x7C)
    INVALID_VALUE, // input 125 (0x7D)
    INVALID_VALUE, // input 126 (0x7E)
    INVALID_VALUE, // input 127 (0x7F)
    INVALID_VALUE, // input 128 (0x80)
    INVALID_VALUE, // input 129 (0x81)
    INVALID_VALUE, // input 130 (0x82)
    INVALID_VALUE, // input 131 (0x83)
    INVALID_VALUE, // input 132 (0x84)
    INVALID_VALUE, // input 133 (0x85)
    INVALID_VALUE, // input 134 (0x86)
    INVALID_VALUE, // input 135 (0x87)
    INVALID_VALUE, // input 136 (0x88)
    INVALID_VALUE, // input 137 (0x89)
    INVALID_VALUE, // input 138 (0x8A)
    INVALID_VALUE, // input 139 (0x8B)
    INVALID_VALUE, // input 140 (0x8C)
    INVALID_VALUE, // input 141 (0x8D)
    INVALID_VALUE, // input 142 (0x8E)
    INVALID_VALUE, // input 143 (0x8F)
    INVALID_VALUE, // input 144 (0x90)
    INVALID_VALUE, // input 145 (0x91)
    INVALID_VALUE, // input 146 (0x92)
    INVALID_VALUE, // input 147 (0x93)
    INVALID_VALUE, // input 148 (0x94)
    INVALID_VALUE, // input 149 (0x95)
    INVALID_VALUE, // input 150 (0x96)
    INVALID_VALUE, // input 151 (0x97)
    INVALID_VALUE, // input 152 (0x98)
    INVALID_VALUE, // input 153 (0x99)
    INVALID_VALUE, // input 154 (0x9A)
    INVALID_VALUE, // input 155 (0x9B)
    INVALID_VALUE, // input 156 (0x9C)
    INVALID_VALUE, // input 157 (0x9D)
    INVALID_VALUE, // input 158 (0x9E)
    INVALID_VALUE, // input 159 (0x9F)
    INVALID_VALUE, // input 160 (0xA0)
    INVALID_VALUE, // input 161 (0xA1)
    INVALID_VALUE, // input 162 (0xA2)
    INVALID_VALUE, // input 163 (0xA3)
    INVALID_VALUE, // input 164 (0xA4)
    INVALID_VALUE, // input 165 (0xA5)
    INVALID_VALUE, // input 166 (0xA6)
    INVALID_VALUE, // input 167 (0xA7)
    INVALID_VALUE, // input 168 (0xA8)
    INVALID_VALUE, // input 169 (0xA9)
    INVALID_VALUE, // input 170 (0xAA)
    INVALID_VALUE, // input 171 (0xAB)
    INVALID_VALUE, // input 172 (0xAC)
    INVALID_VALUE, // input 173 (0xAD)
    INVALID_VALUE, // input 174 (0xAE)
    INVALID_VALUE, // input 175 (0xAF)
    INVALID_VALUE, // input 176 (0xB0)
    INVALID_VALUE, // input 177 (0xB1)
    INVALID_VALUE, // input 178 (0xB2)
    INVALID_VALUE, // input 179 (0xB3)
    INVALID_VALUE, // input 180 (0xB4)
    INVALID_VALUE, // input 181 (0xB5)
    INVALID_VALUE, // input 182 (0xB6)
    INVALID_VALUE, // input 183 (0xB7)
    INVALID_VALUE, // input 184 (0xB8)
    INVALID_VALUE, // input 185 (0xB9)
    INVALID_VALUE, // input 186 (0xBA)
    INVALID_VALUE, // input 187 (0xBB)
    INVALID_VALUE, // input 188 (0xBC)
    INVALID_VALUE, // input 189 (0xBD)
    INVALID_VALUE, // input 190 (0xBE)
    INVALID_VALUE, // input 191 (0xBF)
    INVALID_VALUE, // input 192 (0xC0)
    INVALID_VALUE, // input 193 (0xC1)
    INVALID_VALUE, // input 194 (0xC2)
    INVALID_VALUE, // input 195 (0xC3)
    INVALID_VALUE, // input 196 (0xC4)
    INVALID_VALUE, // input 197 (0xC5)
    INVALID_VALUE, // input 198 (0xC6)
    INVALID_VALUE, // input 199 (0xC7)
    INVALID_VALUE, // input 200 (0xC8)
    INVALID_VALUE, // input 201 (0xC9)
    INVALID_VALUE, // input 202 (0xCA)
    INVALID_VALUE, // input 203 (0xCB)
    INVALID_VALUE, // input 204 (0xCC)
    INVALID_VALUE, // input 205 (0xCD)
    INVALID_VALUE, // input 206 (0xCE)
    INVALID_VALUE, // input 207 (0xCF)
    INVALID_VALUE, // input 208 (0xD0)
    INVALID_VALUE, // input 209 (0xD1)
    INVALID_VALUE, // input 210 (0xD2)
    INVALID_VALUE, // input 211 (0xD3)
    INVALID_VALUE, // input 212 (0xD4)
    INVALID_VALUE, // input 213 (0xD5)
    INVALID_VALUE, // input 214 (0xD6)
    INVALID_VALUE, // input 215 (0xD7)
    INVALID_VALUE, // input 216 (0xD8)
    INVALID_VALUE, // input 217 (0xD9)
    INVALID_VALUE, // input 218 (0xDA)
    INVALID_VALUE, // input 219 (0xDB)
    INVALID_VALUE, // input 220 (0xDC)
    INVALID_VALUE, // input 221 (0xDD)
    INVALID_VALUE, // input 222 (0xDE)
    INVALID_VALUE, // input 223 (0xDF)
    INVALID_VALUE, // input 224 (0xE0)
    INVALID_VALUE, // input 225 (0xE1)
    INVALID_VALUE, // input 226 (0xE2)
    INVALID_VALUE, // input 227 (0xE3)
    INVALID_VALUE, // input 228 (0xE4)
    INVALID_VALUE, // input 229 (0xE5)
    INVALID_VALUE, // input 230 (0xE6)
    INVALID_VALUE, // input 231 (0xE7)
    INVALID_VALUE, // input 232 (0xE8)
    INVALID_VALUE, // input 233 (0xE9)
    INVALID_VALUE, // input 234 (0xEA)
    INVALID_VALUE, // input 235 (0xEB)
    INVALID_VALUE, // input 236 (0xEC)
    INVALID_VALUE, // input 237 (0xED)
    INVALID_VALUE, // input 238 (0xEE)
    INVALID_VALUE, // input 239 (0xEF)
    INVALID_VALUE, // input 240 (0xF0)
    INVALID_VALUE, // input 241 (0xF1)
    INVALID_VALUE, // input 242 (0xF2)
    INVALID_VALUE, // input 243 (0xF3)
    INVALID_VALUE, // input 244 (0xF4)
    INVALID_VALUE, // input 245 (0xF5)
    INVALID_VALUE, // input 246 (0xF6)
    INVALID_VALUE, // input 247 (0xF7)
    INVALID_VALUE, // input 248 (0xF8)
    INVALID_VALUE, // input 249 (0xF9)
    INVALID_VALUE, // input 250 (0xFA)
    INVALID_VALUE, // input 251 (0xFB)
    INVALID_VALUE, // input 252 (0xFC)
    INVALID_VALUE, // input 253 (0xFD)
    INVALID_VALUE, // input 254 (0xFE)
    INVALID_VALUE, // input 255 (0xFF)
];
#[rustfmt::skip]
 const CRYPT_ENCODE: &[u8; 64] = &[
    46, // input 0 (0x0) => '.' (0x2E)
    47, // input 1 (0x1) => '/' (0x2F)
    48, // input 2 (0x2) => '0' (0x30)
    49, // input 3 (0x3) => '1' (0x31)
    50, // input 4 (0x4) => '2' (0x32)
    51, // input 5 (0x5) => '3' (0x33)
    52, // input 6 (0x6) => '4' (0x34)
    53, // input 7 (0x7) => '5' (0x35)
    54, // input 8 (0x8) => '6' (0x36)
    55, // input 9 (0x9) => '7' (0x37)
    56, // input 10 (0xA) => '8' (0x38)
    57, // input 11 (0xB) => '9' (0x39)
    65, // input 12 (0xC) => 'A' (0x41)
    66, // input 13 (0xD) => 'B' (0x42)
    67, // input 14 (0xE) => 'C' (0x43)
    68, // input 15 (0xF) => 'D' (0x44)
    69, // input 16 (0x10) => 'E' (0x45)
    70, // input 17 (0x11) => 'F' (0x46)
    71, // input 18 (0x12) => 'G' (0x47)
    72, // input 19 (0x13) => 'H' (0x48)
    73, // input 20 (0x14) => 'I' (0x49)
    74, // input 21 (0x15) => 'J' (0x4A)
    75, // input 22 (0x16) => 'K' (0x4B)
    76, // input 23 (0x17) => 'L' (0x4C)
    77, // input 24 (0x18) => 'M' (0x4D)
    78, // input 25 (0x19) => 'N' (0x4E)
    79, // input 26 (0x1A) => 'O' (0x4F)
    80, // input 27 (0x1B) => 'P' (0x50)
    81, // input 28 (0x1C) => 'Q' (0x51)
    82, // input 29 (0x1D) => 'R' (0x52)
    83, // input 30 (0x1E) => 'S' (0x53)
    84, // input 31 (0x1F) => 'T' (0x54)
    85, // input 32 (0x20) => 'U' (0x55)
    86, // input 33 (0x21) => 'V' (0x56)
    87, // input 34 (0x22) => 'W' (0x57)
    88, // input 35 (0x23) => 'X' (0x58)
    89, // input 36 (0x24) => 'Y' (0x59)
    90, // input 37 (0x25) => 'Z' (0x5A)
    97, // input 38 (0x26) => 'a' (0x61)
    98, // input 39 (0x27) => 'b' (0x62)
    99, // input 40 (0x28) => 'c' (0x63)
    100, // input 41 (0x29) => 'd' (0x64)
    101, // input 42 (0x2A) => 'e' (0x65)
    102, // input 43 (0x2B) => 'f' (0x66)
    103, // input 44 (0x2C) => 'g' (0x67)
    104, // input 45 (0x2D) => 'h' (0x68)
    105, // input 46 (0x2E) => 'i' (0x69)
    106, // input 47 (0x2F) => 'j' (0x6A)
    107, // input 48 (0x30) => 'k' (0x6B)
    108, // input 49 (0x31) => 'l' (0x6C)
    109, // input 50 (0x32) => 'm' (0x6D)
    110, // input 51 (0x33) => 'n' (0x6E)
    111, // input 52 (0x34) => 'o' (0x6F)
    112, // input 53 (0x35) => 'p' (0x70)
    113, // input 54 (0x36) => 'q' (0x71)
    114, // input 55 (0x37) => 'r' (0x72)
    115, // input 56 (0x38) => 's' (0x73)
    116, // input 57 (0x39) => 't' (0x74)
    117, // input 58 (0x3A) => 'u' (0x75)
    118, // input 59 (0x3B) => 'v' (0x76)
    119, // input 60 (0x3C) => 'w' (0x77)
    120, // input 61 (0x3D) => 'x' (0x78)
    121, // input 62 (0x3E) => 'y' (0x79)
    122, // input 63 (0x3F) => 'z' (0x7A)
];
#[rustfmt::skip]
 const CRYPT_DECODE: &[u8; 256] = &[
    INVALID_VALUE, // input 0 (0x0)
    INVALID_VALUE, // input 1 (0x1)
    INVALID_VALUE, // input 2 (0x2)
    INVALID_VALUE, // input 3 (0x3)
    INVALID_VALUE, // input 4 (0x4)
    INVALID_VALUE, // input 5 (0x5)
    INVALID_VALUE, // input 6 (0x6)
    INVALID_VALUE, // input 7 (0x7)
    INVALID_VALUE, // input 8 (0x8)
    INVALID_VALUE, // input 9 (0x9)
    INVALID_VALUE, // input 10 (0xA)
    INVALID_VALUE, // input 11 (0xB)
    INVALID_VALUE, // input 12 (0xC)
    INVALID_VALUE, // input 13 (0xD)
    INVALID_VALUE, // input 14 (0xE)
    INVALID_VALUE, // input 15 (0xF)
    INVALID_VALUE, // input 16 (0x10)
    INVALID_VALUE, // input 17 (0x11)
    INVALID_VALUE, // input 18 (0x12)
    INVALID_VALUE, // input 19 (0x13)
    INVALID_VALUE, // input 20 (0x14)
    INVALID_VALUE, // input 21 (0x15)
    INVALID_VALUE, // input 22 (0x16)
    INVALID_VALUE, // input 23 (0x17)
    INVALID_VALUE, // input 24 (0x18)
    INVALID_VALUE, // input 25 (0x19)
    INVALID_VALUE, // input 26 (0x1A)
    INVALID_VALUE, // input 27 (0x1B)
    INVALID_VALUE, // input 28 (0x1C)
    INVALID_VALUE, // input 29 (0x1D)
    INVALID_VALUE, // input 30 (0x1E)
    INVALID_VALUE, // input 31 (0x1F)
    INVALID_VALUE, // input 32 (0x20)
    INVALID_VALUE, // input 33 (0x21)
    INVALID_VALUE, // input 34 (0x22)
    INVALID_VALUE, // input 35 (0x23)
    INVALID_VALUE, // input 36 (0x24)
    INVALID_VALUE, // input 37 (0x25)
    INVALID_VALUE, // input 38 (0x26)
    INVALID_VALUE, // input 39 (0x27)
    INVALID_VALUE, // input 40 (0x28)
    INVALID_VALUE, // input 41 (0x29)
    INVALID_VALUE, // input 42 (0x2A)
    INVALID_VALUE, // input 43 (0x2B)
    INVALID_VALUE, // input 44 (0x2C)
    INVALID_VALUE, // input 45 (0x2D)
    0, // input 46 (0x2E char '.') => 0 (0x0)
    1, // input 47 (0x2F char '/') => 1 (0x1)
    2, // input 48 (0x30 char '0') => 2 (0x2)
    3, // input 49 (0x31 char '1') => 3 (0x3)
    4, // input 50 (0x32 char '2') => 4 (0x4)
    5, // input 51 (0x33 char '3') => 5 (0x5)
    6, // input 52 (0x34 char '4') => 6 (0x6)
    7, // input 53 (0x35 char '5') => 7 (0x7)
    8, // input 54 (0x36 char '6') => 8 (0x8)
    9, // input 55 (0x37 char '7') => 9 (0x9)
    10, // input 56 (0x38 char '8') => 10 (0xA)
    11, // input 57 (0x39 char '9') => 11 (0xB)
    INVALID_VALUE, // input 58 (0x3A)
    INVALID_VALUE, // input 59 (0x3B)
    INVALID_VALUE, // input 60 (0x3C)
    INVALID_VALUE, // input 61 (0x3D)
    INVALID_VALUE, // input 62 (0x3E)
    INVALID_VALUE, // input 63 (0x3F)
    INVALID_VALUE, // input 64 (0x40)
    12, // input 65 (0x41 char 'A') => 12 (0xC)
    13, // input 66 (0x42 char 'B') => 13 (0xD)
    14, // input 67 (0x43 char 'C') => 14 (0xE)
    15, // input 68 (0x44 char 'D') => 15 (0xF)
    16, // input 69 (0x45 char 'E') => 16 (0x10)
    17, // input 70 (0x46 char 'F') => 17 (0x11)
    18, // input 71 (0x47 char 'G') => 18 (0x12)
    19, // input 72 (0x48 char 'H') => 19 (0x13)
    20, // input 73 (0x49 char 'I') => 20 (0x14)
    21, // input 74 (0x4A char 'J') => 21 (0x15)
    22, // input 75 (0x4B char 'K') => 22 (0x16)
    23, // input 76 (0x4C char 'L') => 23 (0x17)
    24, // input 77 (0x4D char 'M') => 24 (0x18)
    25, // input 78 (0x4E char 'N') => 25 (0x19)
    26, // input 79 (0x4F char 'O') => 26 (0x1A)
    27, // input 80 (0x50 char 'P') => 27 (0x1B)
    28, // input 81 (0x51 char 'Q') => 28 (0x1C)
    29, // input 82 (0x52 char 'R') => 29 (0x1D)
    30, // input 83 (0x53 char 'S') => 30 (0x1E)
    31, // input 84 (0x54 char 'T') => 31 (0x1F)
    32, // input 85 (0x55 char 'U') => 32 (0x20)
    33, // input 86 (0x56 char 'V') => 33 (0x21)
    34, // input 87 (0x57 char 'W') => 34 (0x22)
    35, // input 88 (0x58 char 'X') => 35 (0x23)
    36, // input 89 (0x59 char 'Y') => 36 (0x24)
    37, // input 90 (0x5A char 'Z') => 37 (0x25)
    INVALID_VALUE, // input 91 (0x5B)
    INVALID_VALUE, // input 92 (0x5C)
    INVALID_VALUE, // input 93 (0x5D)
    INVALID_VALUE, // input 94 (0x5E)
    INVALID_VALUE, // input 95 (0x5F)
    INVALID_VALUE, // input 96 (0x60)
    38, // input 97 (0x61 char 'a') => 38 (0x26)
    39, // input 98 (0x62 char 'b') => 39 (0x27)
    40, // input 99 (0x63 char 'c') => 40 (0x28)
    41, // input 100 (0x64 char 'd') => 41 (0x29)
    42, // input 101 (0x65 char 'e') => 42 (0x2A)
    43, // input 102 (0x66 char 'f') => 43 (0x2B)
    44, // input 103 (0x67 char 'g') => 44 (0x2C)
    45, // input 104 (0x68 char 'h') => 45 (0x2D)
    46, // input 105 (0x69 char 'i') => 46 (0x2E)
    47, // input 106 (0x6A char 'j') => 47 (0x2F)
    48, // input 107 (0x6B char 'k') => 48 (0x30)
    49, // input 108 (0x6C char 'l') => 49 (0x31)
    50, // input 109 (0x6D char 'm') => 50 (0x32)
    51, // input 110 (0x6E char 'n') => 51 (0x33)
    52, // input 111 (0x6F char 'o') => 52 (0x34)
    53, // input 112 (0x70 char 'p') => 53 (0x35)
    54, // input 113 (0x71 char 'q') => 54 (0x36)
    55, // input 114 (0x72 char 'r') => 55 (0x37)
    56, // input 115 (0x73 char 's') => 56 (0x38)
    57, // input 116 (0x74 char 't') => 57 (0x39)
    58, // input 117 (0x75 char 'u') => 58 (0x3A)
    59, // input 118 (0x76 char 'v') => 59 (0x3B)
    60, // input 119 (0x77 char 'w') => 60 (0x3C)
    61, // input 120 (0x78 char 'x') => 61 (0x3D)
    62, // input 121 (0x79 char 'y') => 62 (0x3E)
    63, // input 122 (0x7A char 'z') => 63 (0x3F)
    INVALID_VALUE, // input 123 (0x7B)
    INVALID_VALUE, // input 124 (0x7C)
    INVALID_VALUE, // input 125 (0x7D)
    INVALID_VALUE, // input 126 (0x7E)
    INVALID_VALUE, // input 127 (0x7F)
    INVALID_VALUE, // input 128 (0x80)
    INVALID_VALUE, // input 129 (0x81)
    INVALID_VALUE, // input 130 (0x82)
    INVALID_VALUE, // input 131 (0x83)
    INVALID_VALUE, // input 132 (0x84)
    INVALID_VALUE, // input 133 (0x85)
    INVALID_VALUE, // input 134 (0x86)
    INVALID_VALUE, // input 135 (0x87)
    INVALID_VALUE, // input 136 (0x88)
    INVALID_VALUE, // input 137 (0x89)
    INVALID_VALUE, // input 138 (0x8A)
    INVALID_VALUE, // input 139 (0x8B)
    INVALID_VALUE, // input 140 (0x8C)
    INVALID_VALUE, // input 141 (0x8D)
    INVALID_VALUE, // input 142 (0x8E)
    INVALID_VALUE, // input 143 (0x8F)
    INVALID_VALUE, // input 144 (0x90)
    INVALID_VALUE, // input 145 (0x91)
    INVALID_VALUE, // input 146 (0x92)
    INVALID_VALUE, // input 147 (0x93)
    INVALID_VALUE, // input 148 (0x94)
    INVALID_VALUE, // input 149 (0x95)
    INVALID_VALUE, // input 150 (0x96)
    INVALID_VALUE, // input 151 (0x97)
    INVALID_VALUE, // input 152 (0x98)
    INVALID_VALUE, // input 153 (0x99)
    INVALID_VALUE, // input 154 (0x9A)
    INVALID_VALUE, // input 155 (0x9B)
    INVALID_VALUE, // input 156 (0x9C)
    INVALID_VALUE, // input 157 (0x9D)
    INVALID_VALUE, // input 158 (0x9E)
    INVALID_VALUE, // input 159 (0x9F)
    INVALID_VALUE, // input 160 (0xA0)
    INVALID_VALUE, // input 161 (0xA1)
    INVALID_VALUE, // input 162 (0xA2)
    INVALID_VALUE, // input 163 (0xA3)
    INVALID_VALUE, // input 164 (0xA4)
    INVALID_VALUE, // input 165 (0xA5)
    INVALID_VALUE, // input 166 (0xA6)
    INVALID_VALUE, // input 167 (0xA7)
    INVALID_VALUE, // input 168 (0xA8)
    INVALID_VALUE, // input 169 (0xA9)
    INVALID_VALUE, // input 170 (0xAA)
    INVALID_VALUE, // input 171 (0xAB)
    INVALID_VALUE, // input 172 (0xAC)
    INVALID_VALUE, // input 173 (0xAD)
    INVALID_VALUE, // input 174 (0xAE)
    INVALID_VALUE, // input 175 (0xAF)
    INVALID_VALUE, // input 176 (0xB0)
    INVALID_VALUE, // input 177 (0xB1)
    INVALID_VALUE, // input 178 (0xB2)
    INVALID_VALUE, // input 179 (0xB3)
    INVALID_VALUE, // input 180 (0xB4)
    INVALID_VALUE, // input 181 (0xB5)
    INVALID_VALUE, // input 182 (0xB6)
    INVALID_VALUE, // input 183 (0xB7)
    INVALID_VALUE, // input 184 (0xB8)
    INVALID_VALUE, // input 185 (0xB9)
    INVALID_VALUE, // input 186 (0xBA)
    INVALID_VALUE, // input 187 (0xBB)
    INVALID_VALUE, // input 188 (0xBC)
    INVALID_VALUE, // input 189 (0xBD)
    INVALID_VALUE, // input 190 (0xBE)
    INVALID_VALUE, // input 191 (0xBF)
    INVALID_VALUE, // input 192 (0xC0)
    INVALID_VALUE, // input 193 (0xC1)
    INVALID_VALUE, // input 194 (0xC2)
    INVALID_VALUE, // input 195 (0xC3)
    INVALID_VALUE, // input 196 (0xC4)
    INVALID_VALUE, // input 197 (0xC5)
    INVALID_VALUE, // input 198 (0xC6)
    INVALID_VALUE, // input 199 (0xC7)
    INVALID_VALUE, // input 200 (0xC8)
    INVALID_VALUE, // input 201 (0xC9)
    INVALID_VALUE, // input 202 (0xCA)
    INVALID_VALUE, // input 203 (0xCB)
    INVALID_VALUE, // input 204 (0xCC)
    INVALID_VALUE, // input 205 (0xCD)
    INVALID_VALUE, // input 206 (0xCE)
    INVALID_VALUE, // input 207 (0xCF)
    INVALID_VALUE, // input 208 (0xD0)
    INVALID_VALUE, // input 209 (0xD1)
    INVALID_VALUE, // input 210 (0xD2)
    INVALID_VALUE, // input 211 (0xD3)
    INVALID_VALUE, // input 212 (0xD4)
    INVALID_VALUE, // input 213 (0xD5)
    INVALID_VALUE, // input 214 (0xD6)
    INVALID_VALUE, // input 215 (0xD7)
    INVALID_VALUE, // input 216 (0xD8)
    INVALID_VALUE, // input 217 (0xD9)
    INVALID_VALUE, // input 218 (0xDA)
    INVALID_VALUE, // input 219 (0xDB)
    INVALID_VALUE, // input 220 (0xDC)
    INVALID_VALUE, // input 221 (0xDD)
    INVALID_VALUE, // input 222 (0xDE)
    INVALID_VALUE, // input 223 (0xDF)
    INVALID_VALUE, // input 224 (0xE0)
    INVALID_VALUE, // input 225 (0xE1)
    INVALID_VALUE, // input 226 (0xE2)
    INVALID_VALUE, // input 227 (0xE3)
    INVALID_VALUE, // input 228 (0xE4)
    INVALID_VALUE, // input 229 (0xE5)
    INVALID_VALUE, // input 230 (0xE6)
    INVALID_VALUE, // input 231 (0xE7)
    INVALID_VALUE, // input 232 (0xE8)
    INVALID_VALUE, // input 233 (0xE9)
    INVALID_VALUE, // input 234 (0xEA)
    INVALID_VALUE, // input 235 (0xEB)
    INVALID_VALUE, // input 236 (0xEC)
    INVALID_VALUE, // input 237 (0xED)
    INVALID_VALUE, // input 238 (0xEE)
    INVALID_VALUE, // input 239 (0xEF)
    INVALID_VALUE, // input 240 (0xF0)
    INVALID_VALUE, // input 241 (0xF1)
    INVALID_VALUE, // input 242 (0xF2)
    INVALID_VALUE, // input 243 (0xF3)
    INVALID_VALUE, // input 244 (0xF4)
    INVALID_VALUE, // input 245 (0xF5)
    INVALID_VALUE, // input 246 (0xF6)
    INVALID_VALUE, // input 247 (0xF7)
    INVALID_VALUE, // input 248 (0xF8)
    INVALID_VALUE, // input 249 (0xF9)
    INVALID_VALUE, // input 250 (0xFA)
    INVALID_VALUE, // input 251 (0xFB)
    INVALID_VALUE, // input 252 (0xFC)
    INVALID_VALUE, // input 253 (0xFD)
    INVALID_VALUE, // input 254 (0xFE)
    INVALID_VALUE, // input 255 (0xFF)
];
#[rustfmt::skip]
 const BCRYPT_ENCODE: &[u8; 64] = &[
    46, // input 0 (0x0) => '.' (0x2E)
    47, // input 1 (0x1) => '/' (0x2F)
    65, // input 2 (0x2) => 'A' (0x41)
    66, // input 3 (0x3) => 'B' (0x42)
    67, // input 4 (0x4) => 'C' (0x43)
    68, // input 5 (0x5) => 'D' (0x44)
    69, // input 6 (0x6) => 'E' (0x45)
    70, // input 7 (0x7) => 'F' (0x46)
    71, // input 8 (0x8) => 'G' (0x47)
    72, // input 9 (0x9) => 'H' (0x48)
    73, // input 10 (0xA) => 'I' (0x49)
    74, // input 11 (0xB) => 'J' (0x4A)
    75, // input 12 (0xC) => 'K' (0x4B)
    76, // input 13 (0xD) => 'L' (0x4C)
    77, // input 14 (0xE) => 'M' (0x4D)
    78, // input 15 (0xF) => 'N' (0x4E)
    79, // input 16 (0x10) => 'O' (0x4F)
    80, // input 17 (0x11) => 'P' (0x50)
    81, // input 18 (0x12) => 'Q' (0x51)
    82, // input 19 (0x13) => 'R' (0x52)
    83, // input 20 (0x14) => 'S' (0x53)
    84, // input 21 (0x15) => 'T' (0x54)
    85, // input 22 (0x16) => 'U' (0x55)
    86, // input 23 (0x17) => 'V' (0x56)
    87, // input 24 (0x18) => 'W' (0x57)
    88, // input 25 (0x19) => 'X' (0x58)
    89, // input 26 (0x1A) => 'Y' (0x59)
    90, // input 27 (0x1B) => 'Z' (0x5A)
    97, // input 28 (0x1C) => 'a' (0x61)
    98, // input 29 (0x1D) => 'b' (0x62)
    99, // input 30 (0x1E) => 'c' (0x63)
    100, // input 31 (0x1F) => 'd' (0x64)
    101, // input 32 (0x20) => 'e' (0x65)
    102, // input 33 (0x21) => 'f' (0x66)
    103, // input 34 (0x22) => 'g' (0x67)
    104, // input 35 (0x23) => 'h' (0x68)
    105, // input 36 (0x24) => 'i' (0x69)
    106, // input 37 (0x25) => 'j' (0x6A)
    107, // input 38 (0x26) => 'k' (0x6B)
    108, // input 39 (0x27) => 'l' (0x6C)
    109, // input 40 (0x28) => 'm' (0x6D)
    110, // input 41 (0x29) => 'n' (0x6E)
    111, // input 42 (0x2A) => 'o' (0x6F)
    112, // input 43 (0x2B) => 'p' (0x70)
    113, // input 44 (0x2C) => 'q' (0x71)
    114, // input 45 (0x2D) => 'r' (0x72)
    115, // input 46 (0x2E) => 's' (0x73)
    116, // input 47 (0x2F) => 't' (0x74)
    117, // input 48 (0x30) => 'u' (0x75)
    118, // input 49 (0x31) => 'v' (0x76)
    119, // input 50 (0x32) => 'w' (0x77)
    120, // input 51 (0x33) => 'x' (0x78)
    121, // input 52 (0x34) => 'y' (0x79)
    122, // input 53 (0x35) => 'z' (0x7A)
    48, // input 54 (0x36) => '0' (0x30)
    49, // input 55 (0x37) => '1' (0x31)
    50, // input 56 (0x38) => '2' (0x32)
    51, // input 57 (0x39) => '3' (0x33)
    52, // input 58 (0x3A) => '4' (0x34)
    53, // input 59 (0x3B) => '5' (0x35)
    54, // input 60 (0x3C) => '6' (0x36)
    55, // input 61 (0x3D) => '7' (0x37)
    56, // input 62 (0x3E) => '8' (0x38)
    57, // input 63 (0x3F) => '9' (0x39)
];
#[rustfmt::skip]
 const BCRYPT_DECODE: &[u8; 256] = &[
    INVALID_VALUE, // input 0 (0x0)
    INVALID_VALUE, // input 1 (0x1)
    INVALID_VALUE, // input 2 (0x2)
    INVALID_VALUE, // input 3 (0x3)
    INVALID_VALUE, // input 4 (0x4)
    INVALID_VALUE, // input 5 (0x5)
    INVALID_VALUE, // input 6 (0x6)
    INVALID_VALUE, // input 7 (0x7)
    INVALID_VALUE, // input 8 (0x8)
    INVALID_VALUE, // input 9 (0x9)
    INVALID_VALUE, // input 10 (0xA)
    INVALID_VALUE, // input 11 (0xB)
    INVALID_VALUE, // input 12 (0xC)
    INVALID_VALUE, // input 13 (0xD)
    INVALID_VALUE, // input 14 (0xE)
    INVALID_VALUE, // input 15 (0xF)
    INVALID_VALUE, // input 16 (0x10)
    INVALID_VALUE, // input 17 (0x11)
    INVALID_VALUE, // input 18 (0x12)
    INVALID_VALUE, // input 19 (0x13)
    INVALID_VALUE, // input 20 (0x14)
    INVALID_VALUE, // input 21 (0x15)
    INVALID_VALUE, // input 22 (0x16)
    INVALID_VALUE, // input 23 (0x17)
    INVALID_VALUE, // input 24 (0x18)
    INVALID_VALUE, // input 25 (0x19)
    INVALID_VALUE, // input 26 (0x1A)
    INVALID_VALUE, // input 27 (0x1B)
    INVALID_VALUE, // input 28 (0x1C)
    INVALID_VALUE, // input 29 (0x1D)
    INVALID_VALUE, // input 30 (0x1E)
    INVALID_VALUE, // input 31 (0x1F)
    INVALID_VALUE, // input 32 (0x20)
    INVALID_VALUE, // input 33 (0x21)
    INVALID_VALUE, // input 34 (0x22)
    INVALID_VALUE, // input 35 (0x23)
    INVALID_VALUE, // input 36 (0x24)
    INVALID_VALUE, // input 37 (0x25)
    INVALID_VALUE, // input 38 (0x26)
    INVALID_VALUE, // input 39 (0x27)
    INVALID_VALUE, // input 40 (0x28)
    INVALID_VALUE, // input 41 (0x29)
    INVALID_VALUE, // input 42 (0x2A)
    INVALID_VALUE, // input 43 (0x2B)
    INVALID_VALUE, // input 44 (0x2C)
    INVALID_VALUE, // input 45 (0x2D)
    0, // input 46 (0x2E char '.') => 0 (0x0)
    1, // input 47 (0x2F char '/') => 1 (0x1)
    54, // input 48 (0x30 char '0') => 54 (0x36)
    55, // input 49 (0x31 char '1') => 55 (0x37)
    56, // input 50 (0x32 char '2') => 56 (0x38)
    57, // input 51 (0x33 char '3') => 57 (0x39)
    58, // input 52 (0x34 char '4') => 58 (0x3A)
    59, // input 53 (0x35 char '5') => 59 (0x3B)
    60, // input 54 (0x36 char '6') => 60 (0x3C)
    61, // input 55 (0x37 char '7') => 61 (0x3D)
    62, // input 56 (0x38 char '8') => 62 (0x3E)
    63, // input 57 (0x39 char '9') => 63 (0x3F)
    INVALID_VALUE, // input 58 (0x3A)
    INVALID_VALUE, // input 59 (0x3B)
    INVALID_VALUE, // input 60 (0x3C)
    INVALID_VALUE, // input 61 (0x3D)
    INVALID_VALUE, // input 62 (0x3E)
    INVALID_VALUE, // input 63 (0x3F)
    INVALID_VALUE, // input 64 (0x40)
    2, // input 65 (0x41 char 'A') => 2 (0x2)
    3, // input 66 (0x42 char 'B') => 3 (0x3)
    4, // input 67 (0x43 char 'C') => 4 (0x4)
    5, // input 68 (0x44 char 'D') => 5 (0x5)
    6, // input 69 (0x45 char 'E') => 6 (0x6)
    7, // input 70 (0x46 char 'F') => 7 (0x7)
    8, // input 71 (0x47 char 'G') => 8 (0x8)
    9, // input 72 (0x48 char 'H') => 9 (0x9)
    10, // input 73 (0x49 char 'I') => 10 (0xA)
    11, // input 74 (0x4A char 'J') => 11 (0xB)
    12, // input 75 (0x4B char 'K') => 12 (0xC)
    13, // input 76 (0x4C char 'L') => 13 (0xD)
    14, // input 77 (0x4D char 'M') => 14 (0xE)
    15, // input 78 (0x4E char 'N') => 15 (0xF)
    16, // input 79 (0x4F char 'O') => 16 (0x10)
    17, // input 80 (0x50 char 'P') => 17 (0x11)
    18, // input 81 (0x51 char 'Q') => 18 (0x12)
    19, // input 82 (0x52 char 'R') => 19 (0x13)
    20, // input 83 (0x53 char 'S') => 20 (0x14)
    21, // input 84 (0x54 char 'T') => 21 (0x15)
    22, // input 85 (0x55 char 'U') => 22 (0x16)
    23, // input 86 (0x56 char 'V') => 23 (0x17)
    24, // input 87 (0x57 char 'W') => 24 (0x18)
    25, // input 88 (0x58 char 'X') => 25 (0x19)
    26, // input 89 (0x59 char 'Y') => 26 (0x1A)
    27, // input 90 (0x5A char 'Z') => 27 (0x1B)
    INVALID_VALUE, // input 91 (0x5B)
    INVALID_VALUE, // input 92 (0x5C)
    INVALID_VALUE, // input 93 (0x5D)
    INVALID_VALUE, // input 94 (0x5E)
    INVALID_VALUE, // input 95 (0x5F)
    INVALID_VALUE, // input 96 (0x60)
    28, // input 97 (0x61 char 'a') => 28 (0x1C)
    29, // input 98 (0x62 char 'b') => 29 (0x1D)
    30, // input 99 (0x63 char 'c') => 30 (0x1E)
    31, // input 100 (0x64 char 'd') => 31 (0x1F)
    32, // input 101 (0x65 char 'e') => 32 (0x20)
    33, // input 102 (0x66 char 'f') => 33 (0x21)
    34, // input 103 (0x67 char 'g') => 34 (0x22)
    35, // input 104 (0x68 char 'h') => 35 (0x23)
    36, // input 105 (0x69 char 'i') => 36 (0x24)
    37, // input 106 (0x6A char 'j') => 37 (0x25)
    38, // input 107 (0x6B char 'k') => 38 (0x26)
    39, // input 108 (0x6C char 'l') => 39 (0x27)
    40, // input 109 (0x6D char 'm') => 40 (0x28)
    41, // input 110 (0x6E char 'n') => 41 (0x29)
    42, // input 111 (0x6F char 'o') => 42 (0x2A)
    43, // input 112 (0x70 char 'p') => 43 (0x2B)
    44, // input 113 (0x71 char 'q') => 44 (0x2C)
    45, // input 114 (0x72 char 'r') => 45 (0x2D)
    46, // input 115 (0x73 char 's') => 46 (0x2E)
    47, // input 116 (0x74 char 't') => 47 (0x2F)
    48, // input 117 (0x75 char 'u') => 48 (0x30)
    49, // input 118 (0x76 char 'v') => 49 (0x31)
    50, // input 119 (0x77 char 'w') => 50 (0x32)
    51, // input 120 (0x78 char 'x') => 51 (0x33)
    52, // input 121 (0x79 char 'y') => 52 (0x34)
    53, // input 122 (0x7A char 'z') => 53 (0x35)
    INVALID_VALUE, // input 123 (0x7B)
    INVALID_VALUE, // input 124 (0x7C)
    INVALID_VALUE, // input 125 (0x7D)
    INVALID_VALUE, // input 126 (0x7E)
    INVALID_VALUE, // input 127 (0x7F)
    INVALID_VALUE, // input 128 (0x80)
    INVALID_VALUE, // input 129 (0x81)
    INVALID_VALUE, // input 130 (0x82)
    INVALID_VALUE, // input 131 (0x83)
    INVALID_VALUE, // input 132 (0x84)
    INVALID_VALUE, // input 133 (0x85)
    INVALID_VALUE, // input 134 (0x86)
    INVALID_VALUE, // input 135 (0x87)
    INVALID_VALUE, // input 136 (0x88)
    INVALID_VALUE, // input 137 (0x89)
    INVALID_VALUE, // input 138 (0x8A)
    INVALID_VALUE, // input 139 (0x8B)
    INVALID_VALUE, // input 140 (0x8C)
    INVALID_VALUE, // input 141 (0x8D)
    INVALID_VALUE, // input 142 (0x8E)
    INVALID_VALUE, // input 143 (0x8F)
    INVALID_VALUE, // input 144 (0x90)
    INVALID_VALUE, // input 145 (0x91)
    INVALID_VALUE, // input 146 (0x92)
    INVALID_VALUE, // input 147 (0x93)
    INVALID_VALUE, // input 148 (0x94)
    INVALID_VALUE, // input 149 (0x95)
    INVALID_VALUE, // input 150 (0x96)
    INVALID_VALUE, // input 151 (0x97)
    INVALID_VALUE, // input 152 (0x98)
    INVALID_VALUE, // input 153 (0x99)
    INVALID_VALUE, // input 154 (0x9A)
    INVALID_VALUE, // input 155 (0x9B)
    INVALID_VALUE, // input 156 (0x9C)
    INVALID_VALUE, // input 157 (0x9D)
    INVALID_VALUE, // input 158 (0x9E)
    INVALID_VALUE, // input 159 (0x9F)
    INVALID_VALUE, // input 160 (0xA0)
    INVALID_VALUE, // input 161 (0xA1)
    INVALID_VALUE, // input 162 (0xA2)
    INVALID_VALUE, // input 163 (0xA3)
    INVALID_VALUE, // input 164 (0xA4)
    INVALID_VALUE, // input 165 (0xA5)
    INVALID_VALUE, // input 166 (0xA6)
    INVALID_VALUE, // input 167 (0xA7)
    INVALID_VALUE, // input 168 (0xA8)
    INVALID_VALUE, // input 169 (0xA9)
    INVALID_VALUE, // input 170 (0xAA)
    INVALID_VALUE, // input 171 (0xAB)
    INVALID_VALUE, // input 172 (0xAC)
    INVALID_VALUE, // input 173 (0xAD)
    INVALID_VALUE, // input 174 (0xAE)
    INVALID_VALUE, // input 175 (0xAF)
    INVALID_VALUE, // input 176 (0xB0)
    INVALID_VALUE, // input 177 (0xB1)
    INVALID_VALUE, // input 178 (0xB2)
    INVALID_VALUE, // input 179 (0xB3)
    INVALID_VALUE, // input 180 (0xB4)
    INVALID_VALUE, // input 181 (0xB5)
    INVALID_VALUE, // input 182 (0xB6)
    INVALID_VALUE, // input 183 (0xB7)
    INVALID_VALUE, // input 184 (0xB8)
    INVALID_VALUE, // input 185 (0xB9)
    INVALID_VALUE, // input 186 (0xBA)
    INVALID_VALUE, // input 187 (0xBB)
    INVALID_VALUE, // input 188 (0xBC)
    INVALID_VALUE, // input 189 (0xBD)
    INVALID_VALUE, // input 190 (0xBE)
    INVALID_VALUE, // input 191 (0xBF)
    INVALID_VALUE, // input 192 (0xC0)
    INVALID_VALUE, // input 193 (0xC1)
    INVALID_VALUE, // input 194 (0xC2)
    INVALID_VALUE, // input 195 (0xC3)
    INVALID_VALUE, // input 196 (0xC4)
    INVALID_VALUE, // input 197 (0xC5)
    INVALID_VALUE, // input 198 (0xC6)
    INVALID_VALUE, // input 199 (0xC7)
    INVALID_VALUE, // input 200 (0xC8)
    INVALID_VALUE, // input 201 (0xC9)
    INVALID_VALUE, // input 202 (0xCA)
    INVALID_VALUE, // input 203 (0xCB)
    INVALID_VALUE, // input 204 (0xCC)
    INVALID_VALUE, // input 205 (0xCD)
    INVALID_VALUE, // input 206 (0xCE)
    INVALID_VALUE, // input 207 (0xCF)
    INVALID_VALUE, // input 208 (0xD0)
    INVALID_VALUE, // input 209 (0xD1)
    INVALID_VALUE, // input 210 (0xD2)
    INVALID_VALUE, // input 211 (0xD3)
    INVALID_VALUE, // input 212 (0xD4)
    INVALID_VALUE, // input 213 (0xD5)
    INVALID_VALUE, // input 214 (0xD6)
    INVALID_VALUE, // input 215 (0xD7)
    INVALID_VALUE, // input 216 (0xD8)
    INVALID_VALUE, // input 217 (0xD9)
    INVALID_VALUE, // input 218 (0xDA)
    INVALID_VALUE, // input 219 (0xDB)
    INVALID_VALUE, // input 220 (0xDC)
    INVALID_VALUE, // input 221 (0xDD)
    INVALID_VALUE, // input 222 (0xDE)
    INVALID_VALUE, // input 223 (0xDF)
    INVALID_VALUE, // input 224 (0xE0)
    INVALID_VALUE, // input 225 (0xE1)
    INVALID_VALUE, // input 226 (0xE2)
    INVALID_VALUE, // input 227 (0xE3)
    INVALID_VALUE, // input 228 (0xE4)
    INVALID_VALUE, // input 229 (0xE5)
    INVALID_VALUE, // input 230 (0xE6)
    INVALID_VALUE, // input 231 (0xE7)
    INVALID_VALUE, // input 232 (0xE8)
    INVALID_VALUE, // input 233 (0xE9)
    INVALID_VALUE, // input 234 (0xEA)
    INVALID_VALUE, // input 235 (0xEB)
    INVALID_VALUE, // input 236 (0xEC)
    INVALID_VALUE, // input 237 (0xED)
    INVALID_VALUE, // input 238 (0xEE)
    INVALID_VALUE, // input 239 (0xEF)
    INVALID_VALUE, // input 240 (0xF0)
    INVALID_VALUE, // input 241 (0xF1)
    INVALID_VALUE, // input 242 (0xF2)
    INVALID_VALUE, // input 243 (0xF3)
    INVALID_VALUE, // input 244 (0xF4)
    INVALID_VALUE, // input 245 (0xF5)
    INVALID_VALUE, // input 246 (0xF6)
    INVALID_VALUE, // input 247 (0xF7)
    INVALID_VALUE, // input 248 (0xF8)
    INVALID_VALUE, // input 249 (0xF9)
    INVALID_VALUE, // input 250 (0xFA)
    INVALID_VALUE, // input 251 (0xFB)
    INVALID_VALUE, // input 252 (0xFC)
    INVALID_VALUE, // input 253 (0xFD)
    INVALID_VALUE, // input 254 (0xFE)
    INVALID_VALUE, // input 255 (0xFF)
];
#[rustfmt::skip]
 const IMAP_MUTF7_ENCODE: &[u8; 64] = &[
    65, // input 0 (0x0) => 'A' (0x41)
    66, // input 1 (0x1) => 'B' (0x42)
    67, // input 2 (0x2) => 'C' (0x43)
    68, // input 3 (0x3) => 'D' (0x44)
    69, // input 4 (0x4) => 'E' (0x45)
    70, // input 5 (0x5) => 'F' (0x46)
    71, // input 6 (0x6) => 'G' (0x47)
    72, // input 7 (0x7) => 'H' (0x48)
    73, // input 8 (0x8) => 'I' (0x49)
    74, // input 9 (0x9) => 'J' (0x4A)
    75, // input 10 (0xA) => 'K' (0x4B)
    76, // input 11 (0xB) => 'L' (0x4C)
    77, // input 12 (0xC) => 'M' (0x4D)
    78, // input 13 (0xD) => 'N' (0x4E)
    79, // input 14 (0xE) => 'O' (0x4F)
    80, // input 15 (0xF) => 'P' (0x50)
    81, // input 16 (0x10) => 'Q' (0x51)
    82, // input 17 (0x11) => 'R' (0x52)
    83, // input 18 (0x12) => 'S' (0x53)
    84, // input 19 (0x13) => 'T' (0x54)
    85, // input 20 (0x14) => 'U' (0x55)
    86, // input 21 (0x15) => 'V' (0x56)
    87, // input 22 (0x16) => 'W' (0x57)
    88, // input 23 (0x17) => 'X' (0x58)
    89, // input 24 (0x18) => 'Y' (0x59)
    90, // input 25 (0x19) => 'Z' (0x5A)
    97, // input 26 (0x1A) => 'a' (0x61)
    98, // input 27 (0x1B) => 'b' (0x62)
    99, // input 28 (0x1C) => 'c' (0x63)
    100, // input 29 (0x1D) => 'd' (0x64)
    101, // input 30 (0x1E) => 'e' (0x65)
    102, // input 31 (0x1F) => 'f' (0x66)
    103, // input 32 (0x20) => 'g' (0x67)
    104, // input 33 (0x21) => 'h' (0x68)
    105, // input 34 (0x22) => 'i' (0x69)
    106, // input 35 (0x23) => 'j' (0x6A)
    107, // input 36 (0x24) => 'k' (0x6B)
    108, // input 37 (0x25) => 'l' (0x6C)
    109, // input 38 (0x26) => 'm' (0x6D)
    110, // input 39 (0x27) => 'n' (0x6E)
    111, // input 40 (0x28) => 'o' (0x6F)
    112, // input 41 (0x29) => 'p' (0x70)
    113, // input 42 (0x2A) => 'q' (0x71)
    114, // input 43 (0x2B) => 'r' (0x72)
    115, // input 44 (0x2C) => 's' (0x73)
    116, // input 45 (0x2D) => 't' (0x74)
    117, // input 46 (0x2E) => 'u' (0x75)
    118, // input 47 (0x2F) => 'v' (0x76)
    119, // input 48 (0x30) => 'w' (0x77)
    120, // input 49 (0x31) => 'x' (0x78)
    121, // input 50 (0x32) => 'y' (0x79)
    122, // input 51 (0x33) => 'z' (0x7A)
    48, // input 52 (0x34) => '0' (0x30)
    49, // input 53 (0x35) => '1' (0x31)
    50, // input 54 (0x36) => '2' (0x32)
    51, // input 55 (0x37) => '3' (0x33)
    52, // input 56 (0x38) => '4' (0x34)
    53, // input 57 (0x39) => '5' (0x35)
    54, // input 58 (0x3A) => '6' (0x36)
    55, // input 59 (0x3B) => '7' (0x37)
    56, // input 60 (0x3C) => '8' (0x38)
    57, // input 61 (0x3D) => '9' (0x39)
    43, // input 62 (0x3E) => '+' (0x2B)
    44, // input 63 (0x3F) => ',' (0x2C)
];
#[rustfmt::skip]
 const IMAP_MUTF7_DECODE: &[u8; 256] = &[
    INVALID_VALUE, // input 0 (0x0)
    INVALID_VALUE, // input 1 (0x1)
    INVALID_VALUE, // input 2 (0x2)
    INVALID_VALUE, // input 3 (0x3)
    INVALID_VALUE, // input 4 (0x4)
    INVALID_VALUE, // input 5 (0x5)
    INVALID_VALUE, // input 6 (0x6)
    INVALID_VALUE, // input 7 (0x7)
    INVALID_VALUE, // input 8 (0x8)
    INVALID_VALUE, // input 9 (0x9)
    INVALID_VALUE, // input 10 (0xA)
    INVALID_VALUE, // input 11 (0xB)
    INVALID_VALUE, // input 12 (0xC)
    INVALID_VALUE, // input 13 (0xD)
    INVALID_VALUE, // input 14 (0xE)
    INVALID_VALUE, // input 15 (0xF)
    INVALID_VALUE, // input 16 (0x10)
    INVALID_VALUE, // input 17 (0x11)
    INVALID_VALUE, // input 18 (0x12)
    INVALID_VALUE, // input 19 (0x13)
    INVALID_VALUE, // input 20 (0x14)
    INVALID_VALUE, // input 21 (0x15)
    INVALID_VALUE, // input 22 (0x16)
    INVALID_VALUE, // input 23 (0x17)
    INVALID_VALUE, // input 24 (0x18)
    INVALID_VALUE, // input 25 (0x19)
    INVALID_VALUE, // input 26 (0x1A)
    INVALID_VALUE, // input 27 (0x1B)
    INVALID_VALUE, // input 28 (0x1C)
    INVALID_VALUE, // input 29 (0x1D)
    INVALID_VALUE, // input 30 (0x1E)
    INVALID_VALUE, // input 31 (0x1F)
    INVALID_VALUE, // input 32 (0x20)
    INVALID_VALUE, // input 33 (0x21)
    INVALID_VALUE, // input 34 (0x22)
    INVALID_VALUE, // input 35 (0x23)
    INVALID_VALUE, // input 36 (0x24)
    INVALID_VALUE, // input 37 (0x25)
    INVALID_VALUE, // input 38 (0x26)
    INVALID_VALUE, // input 39 (0x27)
    INVALID_VALUE, // input 40 (0x28)
    INVALID_VALUE, // input 41 (0x29)
    INVALID_VALUE, // input 42 (0x2A)
    62, // input 43 (0x2B char '+') => 62 (0x3E)
    63, // input 44 (0x2C char ',') => 63 (0x3F)
    INVALID_VALUE, // input 45 (0x2D)
    INVALID_VALUE, // input 46 (0x2E)
    INVALID_VALUE, // input 47 (0x2F)
    52, // input 48 (0x30 char '0') => 52 (0x34)
    53, // input 49 (0x31 char '1') => 53 (0x35)
    54, // input 50 (0x32 char '2') => 54 (0x36)
    55, // input 51 (0x33 char '3') => 55 (0x37)
    56, // input 52 (0x34 char '4') => 56 (0x38)
    57, // input 53 (0x35 char '5') => 57 (0x39)
    58, // input 54 (0x36 char '6') => 58 (0x3A)
    59, // input 55 (0x37 char '7') => 59 (0x3B)
    60, // input 56 (0x38 char '8') => 60 (0x3C)
    61, // input 57 (0x39 char '9') => 61 (0x3D)
    INVALID_VALUE, // input 58 (0x3A)
    INVALID_VALUE, // input 59 (0x3B)
    INVALID_VALUE, // input 60 (0x3C)
    INVALID_VALUE, // input 61 (0x3D)
    INVALID_VALUE, // input 62 (0x3E)
    INVALID_VALUE, // input 63 (0x3F)
    INVALID_VALUE, // input 64 (0x40)
    0, // input 65 (0x41 char 'A') => 0 (0x0)
    1, // input 66 (0x42 char 'B') => 1 (0x1)
    2, // input 67 (0x43 char 'C') => 2 (0x2)
    3, // input 68 (0x44 char 'D') => 3 (0x3)
    4, // input 69 (0x45 char 'E') => 4 (0x4)
    5, // input 70 (0x46 char 'F') => 5 (0x5)
    6, // input 71 (0x47 char 'G') => 6 (0x6)
    7, // input 72 (0x48 char 'H') => 7 (0x7)
    8, // input 73 (0x49 char 'I') => 8 (0x8)
    9, // input 74 (0x4A char 'J') => 9 (0x9)
    10, // input 75 (0x4B char 'K') => 10 (0xA)
    11, // input 76 (0x4C char 'L') => 11 (0xB)
    12, // input 77 (0x4D char 'M') => 12 (0xC)
    13, // input 78 (0x4E char 'N') => 13 (0xD)
    14, // input 79 (0x4F char 'O') => 14 (0xE)
    15, // input 80 (0x50 char 'P') => 15 (0xF)
    16, // input 81 (0x51 char 'Q') => 16 (0x10)
    17, // input 82 (0x52 char 'R') => 17 (0x11)
    18, // input 83 (0x53 char 'S') => 18 (0x12)
    19, // input 84 (0x54 char 'T') => 19 (0x13)
    20, // input 85 (0x55 char 'U') => 20 (0x14)
    21, // input 86 (0x56 char 'V') => 21 (0x15)
    22, // input 87 (0x57 char 'W') => 22 (0x16)
    23, // input 88 (0x58 char 'X') => 23 (0x17)
    24, // input 89 (0x59 char 'Y') => 24 (0x18)
    25, // input 90 (0x5A char 'Z') => 25 (0x19)
    INVALID_VALUE, // input 91 (0x5B)
    INVALID_VALUE, // input 92 (0x5C)
    INVALID_VALUE, // input 93 (0x5D)
    INVALID_VALUE, // input 94 (0x5E)
    INVALID_VALUE, // input 95 (0x5F)
    INVALID_VALUE, // input 96 (0x60)
    26, // input 97 (0x61 char 'a') => 26 (0x1A)
    27, // input 98 (0x62 char 'b') => 27 (0x1B)
    28, // input 99 (0x63 char 'c') => 28 (0x1C)
    29, // input 100 (0x64 char 'd') => 29 (0x1D)
    30, // input 101 (0x65 char 'e') => 30 (0x1E)
    31, // input 102 (0x66 char 'f') => 31 (0x1F)
    32, // input 103 (0x67 char 'g') => 32 (0x20)
    33, // input 104 (0x68 char 'h') => 33 (0x21)
    34, // input 105 (0x69 char 'i') => 34 (0x22)
    35, // input 106 (0x6A char 'j') => 35 (0x23)
    36, // input 107 (0x6B char 'k') => 36 (0x24)
    37, // input 108 (0x6C char 'l') => 37 (0x25)
    38, // input 109 (0x6D char 'm') => 38 (0x26)
    39, // input 110 (0x6E char 'n') => 39 (0x27)
    40, // input 111 (0x6F char 'o') => 40 (0x28)
    41, // input 112 (0x70 char 'p') => 41 (0x29)
    42, // input 113 (0x71 char 'q') => 42 (0x2A)
    43, // input 114 (0x72 char 'r') => 43 (0x2B)
    44, // input 115 (0x73 char 's') => 44 (0x2C)
    45, // input 116 (0x74 char 't') => 45 (0x2D)
    46, // input 117 (0x75 char 'u') => 46 (0x2E)
    47, // input 118 (0x76 char 'v') => 47 (0x2F)
    48, // input 119 (0x77 char 'w') => 48 (0x30)
    49, // input 120 (0x78 char 'x') => 49 (0x31)
    50, // input 121 (0x79 char 'y') => 50 (0x32)
    51, // input 122 (0x7A char 'z') => 51 (0x33)
    INVALID_VALUE, // input 123 (0x7B)
    INVALID_VALUE, // input 124 (0x7C)
    INVALID_VALUE, // input 125 (0x7D)
    INVALID_VALUE, // input 126 (0x7E)
    INVALID_VALUE, // input 127 (0x7F)
    INVALID_VALUE, // input 128 (0x80)
    INVALID_VALUE, // input 129 (0x81)
    INVALID_VALUE, // input 130 (0x82)
    INVALID_VALUE, // input 131 (0x83)
    INVALID_VALUE, // input 132 (0x84)
    INVALID_VALUE, // input 133 (0x85)
    INVALID_VALUE, // input 134 (0x86)
    INVALID_VALUE, // input 135 (0x87)
    INVALID_VALUE, // input 136 (0x88)
    INVALID_VALUE, // input 137 (0x89)
    INVALID_VALUE, // input 138 (0x8A)
    INVALID_VALUE, // input 139 (0x8B)
    INVALID_VALUE, // input 140 (0x8C)
    INVALID_VALUE, // input 141 (0x8D)
    INVALID_VALUE, // input 142 (0x8E)
    INVALID_VALUE, // input 143 (0x8F)
    INVALID_VALUE, // input 144 (0x90)
    INVALID_VALUE, // input 145 (0x91)
    INVALID_VALUE, // input 146 (0x92)
    INVALID_VALUE, // input 147 (0x93)
    INVALID_VALUE, // input 148 (0x94)
    INVALID_VALUE, // input 149 (0x95)
    INVALID_VALUE, // input 150 (0x96)
    INVALID_VALUE, // input 151 (0x97)
    INVALID_VALUE, // input 152 (0x98)
    INVALID_VALUE, // input 153 (0x99)
    INVALID_VALUE, // input 154 (0x9A)
    INVALID_VALUE, // input 155 (0x9B)
    INVALID_VALUE, // input 156 (0x9C)
    INVALID_VALUE, // input 157 (0x9D)
    INVALID_VALUE, // input 158 (0x9E)
    INVALID_VALUE, // input 159 (0x9F)
    INVALID_VALUE, // input 160 (0xA0)
    INVALID_VALUE, // input 161 (0xA1)
    INVALID_VALUE, // input 162 (0xA2)
    INVALID_VALUE, // input 163 (0xA3)
    INVALID_VALUE, // input 164 (0xA4)
    INVALID_VALUE, // input 165 (0xA5)
    INVALID_VALUE, // input 166 (0xA6)
    INVALID_VALUE, // input 167 (0xA7)
    INVALID_VALUE, // input 168 (0xA8)
    INVALID_VALUE, // input 169 (0xA9)
    INVALID_VALUE, // input 170 (0xAA)
    INVALID_VALUE, // input 171 (0xAB)
    INVALID_VALUE, // input 172 (0xAC)
    INVALID_VALUE, // input 173 (0xAD)
    INVALID_VALUE, // input 174 (0xAE)
    INVALID_VALUE, // input 175 (0xAF)
    INVALID_VALUE, // input 176 (0xB0)
    INVALID_VALUE, // input 177 (0xB1)
    INVALID_VALUE, // input 178 (0xB2)
    INVALID_VALUE, // input 179 (0xB3)
    INVALID_VALUE, // input 180 (0xB4)
    INVALID_VALUE, // input 181 (0xB5)
    INVALID_VALUE, // input 182 (0xB6)
    INVALID_VALUE, // input 183 (0xB7)
    INVALID_VALUE, // input 184 (0xB8)
    INVALID_VALUE, // input 185 (0xB9)
    INVALID_VALUE, // input 186 (0xBA)
    INVALID_VALUE, // input 187 (0xBB)
    INVALID_VALUE, // input 188 (0xBC)
    INVALID_VALUE, // input 189 (0xBD)
    INVALID_VALUE, // input 190 (0xBE)
    INVALID_VALUE, // input 191 (0xBF)
    INVALID_VALUE, // input 192 (0xC0)
    INVALID_VALUE, // input 193 (0xC1)
    INVALID_VALUE, // input 194 (0xC2)
    INVALID_VALUE, // input 195 (0xC3)
    INVALID_VALUE, // input 196 (0xC4)
    INVALID_VALUE, // input 197 (0xC5)
    INVALID_VALUE, // input 198 (0xC6)
    INVALID_VALUE, // input 199 (0xC7)
    INVALID_VALUE, // input 200 (0xC8)
    INVALID_VALUE, // input 201 (0xC9)
    INVALID_VALUE, // input 202 (0xCA)
    INVALID_VALUE, // input 203 (0xCB)
    INVALID_VALUE, // input 204 (0xCC)
    INVALID_VALUE, // input 205 (0xCD)
    INVALID_VALUE, // input 206 (0xCE)
    INVALID_VALUE, // input 207 (0xCF)
    INVALID_VALUE, // input 208 (0xD0)
    INVALID_VALUE, // input 209 (0xD1)
    INVALID_VALUE, // input 210 (0xD2)
    INVALID_VALUE, // input 211 (0xD3)
    INVALID_VALUE, // input 212 (0xD4)
    INVALID_VALUE, // input 213 (0xD5)
    INVALID_VALUE, // input 214 (0xD6)
    INVALID_VALUE, // input 215 (0xD7)
    INVALID_VALUE, // input 216 (0xD8)
    INVALID_VALUE, // input 217 (0xD9)
    INVALID_VALUE, // input 218 (0xDA)
    INVALID_VALUE, // input 219 (0xDB)
    INVALID_VALUE, // input 220 (0xDC)
    INVALID_VALUE, // input 221 (0xDD)
    INVALID_VALUE, // input 222 (0xDE)
    INVALID_VALUE, // input 223 (0xDF)
    INVALID_VALUE, // input 224 (0xE0)
    INVALID_VALUE, // input 225 (0xE1)
    INVALID_VALUE, // input 226 (0xE2)
    INVALID_VALUE, // input 227 (0xE3)
    INVALID_VALUE, // input 228 (0xE4)
    INVALID_VALUE, // input 229 (0xE5)
    INVALID_VALUE, // input 230 (0xE6)
    INVALID_VALUE, // input 231 (0xE7)
    INVALID_VALUE, // input 232 (0xE8)
    INVALID_VALUE, // input 233 (0xE9)
    INVALID_VALUE, // input 234 (0xEA)
    INVALID_VALUE, // input 235 (0xEB)
    INVALID_VALUE, // input 236 (0xEC)
    INVALID_VALUE, // input 237 (0xED)
    INVALID_VALUE, // input 238 (0xEE)
    INVALID_VALUE, // input 239 (0xEF)
    INVALID_VALUE, // input 240 (0xF0)
    INVALID_VALUE, // input 241 (0xF1)
    INVALID_VALUE, // input 242 (0xF2)
    INVALID_VALUE, // input 243 (0xF3)
    INVALID_VALUE, // input 244 (0xF4)
    INVALID_VALUE, // input 245 (0xF5)
    INVALID_VALUE, // input 246 (0xF6)
    INVALID_VALUE, // input 247 (0xF7)
    INVALID_VALUE, // input 248 (0xF8)
    INVALID_VALUE, // input 249 (0xF9)
    INVALID_VALUE, // input 250 (0xFA)
    INVALID_VALUE, // input 251 (0xFB)
    INVALID_VALUE, // input 252 (0xFC)
    INVALID_VALUE, // input 253 (0xFD)
    INVALID_VALUE, // input 254 (0xFE)
    INVALID_VALUE, // input 255 (0xFF)
];
#[rustfmt::skip]
 const BINHEX_ENCODE: &[u8; 64] = &[
    33, // input 0 (0x0) => '!' (0x21)
    34, // input 1 (0x1) => '"' (0x22)
    35, // input 2 (0x2) => '#' (0x23)
    36, // input 3 (0x3) => '$' (0x24)
    37, // input 4 (0x4) => '%' (0x25)
    38, // input 5 (0x5) => '&' (0x26)
    39, // input 6 (0x6) => ''' (0x27)
    40, // input 7 (0x7) => '(' (0x28)
    41, // input 8 (0x8) => ')' (0x29)
    42, // input 9 (0x9) => '*' (0x2A)
    43, // input 10 (0xA) => '+' (0x2B)
    44, // input 11 (0xB) => ',' (0x2C)
    45, // input 12 (0xC) => '-' (0x2D)
    48, // input 13 (0xD) => '0' (0x30)
    49, // input 14 (0xE) => '1' (0x31)
    50, // input 15 (0xF) => '2' (0x32)
    51, // input 16 (0x10) => '3' (0x33)
    52, // input 17 (0x11) => '4' (0x34)
    53, // input 18 (0x12) => '5' (0x35)
    54, // input 19 (0x13) => '6' (0x36)
    55, // input 20 (0x14) => '7' (0x37)
    56, // input 21 (0x15) => '8' (0x38)
    57, // input 22 (0x16) => '9' (0x39)
    64, // input 23 (0x17) => '@' (0x40)
    65, // input 24 (0x18) => 'A' (0x41)
    66, // input 25 (0x19) => 'B' (0x42)
    67, // input 26 (0x1A) => 'C' (0x43)
    68, // input 27 (0x1B) => 'D' (0x44)
    69, // input 28 (0x1C) => 'E' (0x45)
    70, // input 29 (0x1D) => 'F' (0x46)
    71, // input 30 (0x1E) => 'G' (0x47)
    72, // input 31 (0x1F) => 'H' (0x48)
    73, // input 32 (0x20) => 'I' (0x49)
    74, // input 33 (0x21) => 'J' (0x4A)
    75, // input 34 (0x22) => 'K' (0x4B)
    76, // input 35 (0x23) => 'L' (0x4C)
    77, // input 36 (0x24) => 'M' (0x4D)
    78, // input 37 (0x25) => 'N' (0x4E)
    80, // input 38 (0x26) => 'P' (0x50)
    81, // input 39 (0x27) => 'Q' (0x51)
    82, // input 40 (0x28) => 'R' (0x52)
    83, // input 41 (0x29) => 'S' (0x53)
    84, // input 42 (0x2A) => 'T' (0x54)
    85, // input 43 (0x2B) => 'U' (0x55)
    86, // input 44 (0x2C) => 'V' (0x56)
    88, // input 45 (0x2D) => 'X' (0x58)
    89, // input 46 (0x2E) => 'Y' (0x59)
    90, // input 47 (0x2F) => 'Z' (0x5A)
    91, // input 48 (0x30) => '[' (0x5B)
    96, // input 49 (0x31) => '`' (0x60)
    97, // input 50 (0x32) => 'a' (0x61)
    98, // input 51 (0x33) => 'b' (0x62)
    99, // input 52 (0x34) => 'c' (0x63)
    100, // input 53 (0x35) => 'd' (0x64)
    101, // input 54 (0x36) => 'e' (0x65)
    104, // input 55 (0x37) => 'h' (0x68)
    105, // input 56 (0x38) => 'i' (0x69)
    106, // input 57 (0x39) => 'j' (0x6A)
    107, // input 58 (0x3A) => 'k' (0x6B)
    108, // input 59 (0x3B) => 'l' (0x6C)
    109, // input 60 (0x3C) => 'm' (0x6D)
    112, // input 61 (0x3D) => 'p' (0x70)
    113, // input 62 (0x3E) => 'q' (0x71)
    114, // input 63 (0x3F) => 'r' (0x72)
];
#[rustfmt::skip]
 const BINHEX_DECODE: &[u8; 256] = &[
    INVALID_VALUE, // input 0 (0x0)
    INVALID_VALUE, // input 1 (0x1)
    INVALID_VALUE, // input 2 (0x2)
    INVALID_VALUE, // input 3 (0x3)
    INVALID_VALUE, // input 4 (0x4)
    INVALID_VALUE, // input 5 (0x5)
    INVALID_VALUE, // input 6 (0x6)
    INVALID_VALUE, // input 7 (0x7)
    INVALID_VALUE, // input 8 (0x8)
    INVALID_VALUE, // input 9 (0x9)
    INVALID_VALUE, // input 10 (0xA)
    INVALID_VALUE, // input 11 (0xB)
    INVALID_VALUE, // input 12 (0xC)
    INVALID_VALUE, // input 13 (0xD)
    INVALID_VALUE, // input 14 (0xE)
    INVALID_VALUE, // input 15 (0xF)
    INVALID_VALUE, // input 16 (0x10)
    INVALID_VALUE, // input 17 (0x11)
    INVALID_VALUE, // input 18 (0x12)
    INVALID_VALUE, // input 19 (0x13)
    INVALID_VALUE, // input 20 (0x14)
    INVALID_VALUE, // input 21 (0x15)
    INVALID_VALUE, // input 22 (0x16)
    INVALID_VALUE, // input 23 (0x17)
    INVALID_VALUE, // input 24 (0x18)
    INVALID_VALUE, // input 25 (0x19)
    INVALID_VALUE, // input 26 (0x1A)
    INVALID_VALUE, // input 27 (0x1B)
    INVALID_VALUE, // input 28 (0x1C)
    INVALID_VALUE, // input 29 (0x1D)
    INVALID_VALUE, // input 30 (0x1E)
    INVALID_VALUE, // input 31 (0x1F)
    INVALID_VALUE, // input 32 (0x20)
    0, // input 33 (0x21 char '!') => 0 (0x0)
    1, // input 34 (0x22 char '"') => 1 (0x1)
    2, // input 35 (0x23 char '#') => 2 (0x2)
    3, // input 36 (0x24 char '$') => 3 (0x3)
    4, // input 37 (0x25 char '%') => 4 (0x4)
    5, // input 38 (0x26 char '&') => 5 (0x5)
    6, // input 39 (0x27 char ''') => 6 (0x6)
    7, // input 40 (0x28 char '(') => 7 (0x7)
    8, // input 41 (0x29 char ')') => 8 (0x8)
    9, // input 42 (0x2A char '*') => 9 (0x9)
    10, // input 43 (0x2B char '+') => 10 (0xA)
    11, // input 44 (0x2C char ',') => 11 (0xB)
    12, // input 45 (0x2D char '-') => 12 (0xC)
    INVALID_VALUE, // input 46 (0x2E)
    INVALID_VALUE, // input 47 (0x2F)
    13, // input 48 (0x30 char '0') => 13 (0xD)
    14, // input 49 (0x31 char '1') => 14 (0xE)
    15, // input 50 (0x32 char '2') => 15 (0xF)
    16, // input 51 (0x33 char '3') => 16 (0x10)
    17, // input 52 (0x34 char '4') => 17 (0x11)
    18, // input 53 (0x35 char '5') => 18 (0x12)
    19, // input 54 (0x36 char '6') => 19 (0x13)
    20, // input 55 (0x37 char '7') => 20 (0x14)
    21, // input 56 (0x38 char '8') => 21 (0x15)
    22, // input 57 (0x39 char '9') => 22 (0x16)
    INVALID_VALUE, // input 58 (0x3A)
    INVALID_VALUE, // input 59 (0x3B)
    INVALID_VALUE, // input 60 (0x3C)
    INVALID_VALUE, // input 61 (0x3D)
    INVALID_VALUE, // input 62 (0x3E)
    INVALID_VALUE, // input 63 (0x3F)
    23, // input 64 (0x40 char '@') => 23 (0x17)
    24, // input 65 (0x41 char 'A') => 24 (0x18)
    25, // input 66 (0x42 char 'B') => 25 (0x19)
    26, // input 67 (0x43 char 'C') => 26 (0x1A)
    27, // input 68 (0x44 char 'D') => 27 (0x1B)
    28, // input 69 (0x45 char 'E') => 28 (0x1C)
    29, // input 70 (0x46 char 'F') => 29 (0x1D)
    30, // input 71 (0x47 char 'G') => 30 (0x1E)
    31, // input 72 (0x48 char 'H') => 31 (0x1F)
    32, // input 73 (0x49 char 'I') => 32 (0x20)
    33, // input 74 (0x4A char 'J') => 33 (0x21)
    34, // input 75 (0x4B char 'K') => 34 (0x22)
    35, // input 76 (0x4C char 'L') => 35 (0x23)
    36, // input 77 (0x4D char 'M') => 36 (0x24)
    37, // input 78 (0x4E char 'N') => 37 (0x25)
    INVALID_VALUE, // input 79 (0x4F)
    38, // input 80 (0x50 char 'P') => 38 (0x26)
    39, // input 81 (0x51 char 'Q') => 39 (0x27)
    40, // input 82 (0x52 char 'R') => 40 (0x28)
    41, // input 83 (0x53 char 'S') => 41 (0x29)
    42, // input 84 (0x54 char 'T') => 42 (0x2A)
    43, // input 85 (0x55 char 'U') => 43 (0x2B)
    44, // input 86 (0x56 char 'V') => 44 (0x2C)
    INVALID_VALUE, // input 87 (0x57)
    45, // input 88 (0x58 char 'X') => 45 (0x2D)
    46, // input 89 (0x59 char 'Y') => 46 (0x2E)
    47, // input 90 (0x5A char 'Z') => 47 (0x2F)
    48, // input 91 (0x5B char '[') => 48 (0x30)
    INVALID_VALUE, // input 92 (0x5C)
    INVALID_VALUE, // input 93 (0x5D)
    INVALID_VALUE, // input 94 (0x5E)
    INVALID_VALUE, // input 95 (0x5F)
    49, // input 96 (0x60 char '`') => 49 (0x31)
    50, // input 97 (0x61 char 'a') => 50 (0x32)
    51, // input 98 (0x62 char 'b') => 51 (0x33)
    52, // input 99 (0x63 char 'c') => 52 (0x34)
    53, // input 100 (0x64 char 'd') => 53 (0x35)
    54, // input 101 (0x65 char 'e') => 54 (0x36)
    INVALID_VALUE, // input 102 (0x66)
    INVALID_VALUE, // input 103 (0x67)
    55, // input 104 (0x68 char 'h') => 55 (0x37)
    56, // input 105 (0x69 char 'i') => 56 (0x38)
    57, // input 106 (0x6A char 'j') => 57 (0x39)
    58, // input 107 (0x6B char 'k') => 58 (0x3A)
    59, // input 108 (0x6C char 'l') => 59 (0x3B)
    60, // input 109 (0x6D char 'm') => 60 (0x3C)
    INVALID_VALUE, // input 110 (0x6E)
    INVALID_VALUE, // input 111 (0x6F)
    61, // input 112 (0x70 char 'p') => 61 (0x3D)
    62, // input 113 (0x71 char 'q') => 62 (0x3E)
    63, // input 114 (0x72 char 'r') => 63 (0x3F)
    INVALID_VALUE, // input 115 (0x73)
    INVALID_VALUE, // input 116 (0x74)
    INVALID_VALUE, // input 117 (0x75)
    INVALID_VALUE, // input 118 (0x76)
    INVALID_VALUE, // input 119 (0x77)
    INVALID_VALUE, // input 120 (0x78)
    INVALID_VALUE, // input 121 (0x79)
    INVALID_VALUE, // input 122 (0x7A)
    INVALID_VALUE, // input 123 (0x7B)
    INVALID_VALUE, // input 124 (0x7C)
    INVALID_VALUE, // input 125 (0x7D)
    INVALID_VALUE, // input 126 (0x7E)
    INVALID_VALUE, // input 127 (0x7F)
    INVALID_VALUE, // input 128 (0x80)
    INVALID_VALUE, // input 129 (0x81)
    INVALID_VALUE, // input 130 (0x82)
    INVALID_VALUE, // input 131 (0x83)
    INVALID_VALUE, // input 132 (0x84)
    INVALID_VALUE, // input 133 (0x85)
    INVALID_VALUE, // input 134 (0x86)
    INVALID_VALUE, // input 135 (0x87)
    INVALID_VALUE, // input 136 (0x88)
    INVALID_VALUE, // input 137 (0x89)
    INVALID_VALUE, // input 138 (0x8A)
    INVALID_VALUE, // input 139 (0x8B)
    INVALID_VALUE, // input 140 (0x8C)
    INVALID_VALUE, // input 141 (0x8D)
    INVALID_VALUE, // input 142 (0x8E)
    INVALID_VALUE, // input 143 (0x8F)
    INVALID_VALUE, // input 144 (0x90)
    INVALID_VALUE, // input 145 (0x91)
    INVALID_VALUE, // input 146 (0x92)
    INVALID_VALUE, // input 147 (0x93)
    INVALID_VALUE, // input 148 (0x94)
    INVALID_VALUE, // input 149 (0x95)
    INVALID_VALUE, // input 150 (0x96)
    INVALID_VALUE, // input 151 (0x97)
    INVALID_VALUE, // input 152 (0x98)
    INVALID_VALUE, // input 153 (0x99)
    INVALID_VALUE, // input 154 (0x9A)
    INVALID_VALUE, // input 155 (0x9B)
    INVALID_VALUE, // input 156 (0x9C)
    INVALID_VALUE, // input 157 (0x9D)
    INVALID_VALUE, // input 158 (0x9E)
    INVALID_VALUE, // input 159 (0x9F)
    INVALID_VALUE, // input 160 (0xA0)
    INVALID_VALUE, // input 161 (0xA1)
    INVALID_VALUE, // input 162 (0xA2)
    INVALID_VALUE, // input 163 (0xA3)
    INVALID_VALUE, // input 164 (0xA4)
    INVALID_VALUE, // input 165 (0xA5)
    INVALID_VALUE, // input 166 (0xA6)
    INVALID_VALUE, // input 167 (0xA7)
    INVALID_VALUE, // input 168 (0xA8)
    INVALID_VALUE, // input 169 (0xA9)
    INVALID_VALUE, // input 170 (0xAA)
    INVALID_VALUE, // input 171 (0xAB)
    INVALID_VALUE, // input 172 (0xAC)
    INVALID_VALUE, // input 173 (0xAD)
    INVALID_VALUE, // input 174 (0xAE)
    INVALID_VALUE, // input 175 (0xAF)
    INVALID_VALUE, // input 176 (0xB0)
    INVALID_VALUE, // input 177 (0xB1)
    INVALID_VALUE, // input 178 (0xB2)
    INVALID_VALUE, // input 179 (0xB3)
    INVALID_VALUE, // input 180 (0xB4)
    INVALID_VALUE, // input 181 (0xB5)
    INVALID_VALUE, // input 182 (0xB6)
    INVALID_VALUE, // input 183 (0xB7)
    INVALID_VALUE, // input 184 (0xB8)
    INVALID_VALUE, // input 185 (0xB9)
    INVALID_VALUE, // input 186 (0xBA)
    INVALID_VALUE, // input 187 (0xBB)
    INVALID_VALUE, // input 188 (0xBC)
    INVALID_VALUE, // input 189 (0xBD)
    INVALID_VALUE, // input 190 (0xBE)
    INVALID_VALUE, // input 191 (0xBF)
    INVALID_VALUE, // input 192 (0xC0)
    INVALID_VALUE, // input 193 (0xC1)
    INVALID_VALUE, // input 194 (0xC2)
    INVALID_VALUE, // input 195 (0xC3)
    INVALID_VALUE, // input 196 (0xC4)
    INVALID_VALUE, // input 197 (0xC5)
    INVALID_VALUE, // input 198 (0xC6)
    INVALID_VALUE, // input 199 (0xC7)
    INVALID_VALUE, // input 200 (0xC8)
    INVALID_VALUE, // input 201 (0xC9)
    INVALID_VALUE, // input 202 (0xCA)
    INVALID_VALUE, // input 203 (0xCB)
    INVALID_VALUE, // input 204 (0xCC)
    INVALID_VALUE, // input 205 (0xCD)
    INVALID_VALUE, // input 206 (0xCE)
    INVALID_VALUE, // input 207 (0xCF)
    INVALID_VALUE, // input 208 (0xD0)
    INVALID_VALUE, // input 209 (0xD1)
    INVALID_VALUE, // input 210 (0xD2)
    INVALID_VALUE, // input 211 (0xD3)
    INVALID_VALUE, // input 212 (0xD4)
    INVALID_VALUE, // input 213 (0xD5)
    INVALID_VALUE, // input 214 (0xD6)
    INVALID_VALUE, // input 215 (0xD7)
    INVALID_VALUE, // input 216 (0xD8)
    INVALID_VALUE, // input 217 (0xD9)
    INVALID_VALUE, // input 218 (0xDA)
    INVALID_VALUE, // input 219 (0xDB)
    INVALID_VALUE, // input 220 (0xDC)
    INVALID_VALUE, // input 221 (0xDD)
    INVALID_VALUE, // input 222 (0xDE)
    INVALID_VALUE, // input 223 (0xDF)
    INVALID_VALUE, // input 224 (0xE0)
    INVALID_VALUE, // input 225 (0xE1)
    INVALID_VALUE, // input 226 (0xE2)
    INVALID_VALUE, // input 227 (0xE3)
    INVALID_VALUE, // input 228 (0xE4)
    INVALID_VALUE, // input 229 (0xE5)
    INVALID_VALUE, // input 230 (0xE6)
    INVALID_VALUE, // input 231 (0xE7)
    INVALID_VALUE, // input 232 (0xE8)
    INVALID_VALUE, // input 233 (0xE9)
    INVALID_VALUE, // input 234 (0xEA)
    INVALID_VALUE, // input 235 (0xEB)
    INVALID_VALUE, // input 236 (0xEC)
    INVALID_VALUE, // input 237 (0xED)
    INVALID_VALUE, // input 238 (0xEE)
    INVALID_VALUE, // input 239 (0xEF)
    INVALID_VALUE, // input 240 (0xF0)
    INVALID_VALUE, // input 241 (0xF1)
    INVALID_VALUE, // input 242 (0xF2)
    INVALID_VALUE, // input 243 (0xF3)
    INVALID_VALUE, // input 244 (0xF4)
    INVALID_VALUE, // input 245 (0xF5)
    INVALID_VALUE, // input 246 (0xF6)
    INVALID_VALUE, // input 247 (0xF7)
    INVALID_VALUE, // input 248 (0xF8)
    INVALID_VALUE, // input 249 (0xF9)
    INVALID_VALUE, // input 250 (0xFA)
    INVALID_VALUE, // input 251 (0xFB)
    INVALID_VALUE, // input 252 (0xFC)
    INVALID_VALUE, // input 253 (0xFD)
    INVALID_VALUE, // input 254 (0xFE)
    INVALID_VALUE, // input 255 (0xFF)
];
