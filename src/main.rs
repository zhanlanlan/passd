use std::env;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::io::{BufReader, BufWriter, LineWriter, SeekFrom};
use std::path::{Path, PathBuf};

mod encode;
use encode::{decode, encode};

const FILE_NAME: &'static str = ".passd";

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("usage:");
        println!("\tpassd add    'some message'");
        println!("\tpassd list   ");
        println!("\tpassd find   'some keyword'");
        println!("\tpassd delete 'number'");
        return;
    }

    let command: &str = &args[1];

    let home_dir = env::var("HOME").unwrap_or("".to_string());

    let file_path = Path::new(&home_dir).join(FILE_NAME);

    let f = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .open(&file_path)
        .expect(&format!("打开文件 {} 失败", file_path.to_str().unwrap()));

    match command {
        "add" => {
            if args.len() != 3 {
                println!("用法: passd add 'some message'");
                return;
            }
            add(f, &args[2])
        }
        "list" => list(f),
        "find" => {
            if args.len() != 3 {
                println!("用法: passd find 'some keywords'");
                return;
            }
            find(f, &args[2])
        }
        "delete" => delete(),
        _ => println!("为什么要给我一个我不能识别的参数")
    };
}

fn add(mut f: File, content: &str) {
    println!("before encode: {}", content);
    let content = encode(content);
    println!("after encode: {}", content);

    f.seek(SeekFrom::End(0)).expect("写入失败");
    f.write(content.as_bytes()).expect("写入失败");
    f.write(b"\n").expect("写入失败");
}

fn list(mut f: File) {
    f.seek(SeekFrom::Start(0)).expect("读取失败");
    let f = BufReader::new(f);
    for li in f.lines() {
        let content = li.expect("读取失败");
        let content = String::from_utf8(decode(content).expect("读取失败")).expect("读取失败");
        println!("{}", content);
    }
}

fn find(mut f: File, keyword: &str) {
    f.seek(SeekFrom::Start(0)).expect("读取失败");
    let f = BufReader::new(f);
    for li in f.lines() {
        let content = li.expect("读取失败");
        let content = String::from_utf8(decode(content).expect("读取失败")).expect("读取失败");
        if content.contains(keyword) {
            println!("{}", content);
        }
    }
}

fn delete() {
    println!("我正在思考这个功能该怎么做")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_something() {
        println!("{}", encode("input"));
        println!(
            "{}",
            String::from_utf8(decode("5oiR5piv5L2g54i454i4").unwrap()).unwrap()
        );
    }
}
