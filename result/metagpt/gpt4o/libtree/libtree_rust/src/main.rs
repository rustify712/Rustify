use std::env;
use std::ffi::CString;
use std::fs::File;
use std::io::{self, BufRead};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr;
use std::slice;
use std::str;

mod libtree;

use libtree::*;

fn main() {
    let mut s = LibtreeState::new();

    // Enable or disable colors (no-color.com)
    s.color = env::var("NO_COLOR").is_err() && atty::is(atty::Stream::Stdout);
    s.verbosity = 0;
    s.path = false;
    s.max_depth = MAX_RECURSION_DEPTH;

    let mut positional = 1;

    let uname_val = uname::uname().expect("Failed to get system information");

    s.platform = uname_val.machine;
    s.osname = uname_val.sysname;
    s.osrel = uname_val.release;
    s.ld_conf_file = "/etc/ld.so.conf".to_string();

    if uname_val.sysname == "FreeBSD" {
        s.ld_conf_file = "/etc/ld-elf.so.conf".to_string();
    }

    s.lib = "lib".to_string();

    let mut opt_help = false;
    let mut opt_version = false;
    let mut opt_raw = false;

    let args: Vec<String> = env::args().collect();
    let argc = args.len();

    for i in 1..argc {
        let arg = &args[i];

        if opt_raw || !arg.starts_with('-') || arg == "-" {
            args[positional] = arg.clone();
            positional += 1;
            continue;
        }

        let mut arg = &arg[1..];

        if arg.starts_with('-') {
            arg = &arg[1..];

            if arg.is_empty() {
                opt_raw = true;
                continue;
            }

            match arg {
                "version" => opt_version = true,
                "path" => s.path = true,
                "verbose" => s.verbosity += 1,
                "help" => opt_help = true,
                "ldconf" => {
                    if i + 1 == argc {
                        eprintln!("Expected value after `--ldconf`");
                        return;
                    }
                    s.ld_conf_file = args[i + 1].clone();
                }
                "max-depth" => {
                    if i + 1 == argc {
                        eprintln!("Expected value after `--max-depth`");
                        return;
                    }
                    s.max_depth = args[i + 1].parse().unwrap_or(MAX_RECURSION_DEPTH);
                    if s.max_depth > MAX_RECURSION_DEPTH {
                        s.max_depth = MAX_RECURSION_DEPTH;
                    }
                }
                _ => {
                    eprintln!("Unrecognized flag `--{}`", arg);
                    return;
                }
            }

            continue;
        }

        for c in arg.chars() {
            match c {
                'h' => opt_help = true,
                'p' => s.path = true,
                'v' => s.verbosity += 1,
                _ => {
                    eprintln!("Unrecognized flag `-{}`", c);
                    return;
                }
            }
        }
    }

    args.remove(0);
    positional -= 1;

    if opt_help || (!opt_version && positional == 0) {
        print_help(&s);
        return;
    }

    if opt_version {
        println!("{}", VERSION);
        return;
    }

    let exit_code = print_tree(positional, &args, &mut s);
    std::process::exit(exit_code);
}

fn print_help(s: &LibtreeState) {
    println!(
        "Show the dynamic dependency tree of ELF files\n\
         Usage: libtree [OPTION]... [--] FILE [FILES]...\n\
         \n\
         -h, --help     Print help info\n\
         --version  Print version info\n\
         \n\
         File names starting with '-', for example '-.so', can be specified as follows:\n\
         libtree -- -.so\n\
         \n\
         Locating libs options:\n\
         -p, --path       Show the path of libraries instead of the soname\n\
         -v               Show libraries skipped by default*\n\
         -vv              Show dependencies of libraries skipped by default*\n\
         -vvv             Show dependencies of already encountered libraries\n\
         --ldconf <path>  Config file for extra search paths [{}]\n\
         --max-depth <n>  Limit library traversal to at most n levels of depth\n\
         \n\
         * For brevity, the following libraries are not shown by default:\n\
         ",
        s.ld_conf_file
    );

    let num_excluded = EXCLUDE_LIST.len();
    let mut cursor_x = 3;
    for (j, &lib) in EXCLUDE_LIST.iter().enumerate() {
        cursor_x += lib.len();
        if cursor_x > 60 {
            cursor_x = 3;
            print!("\n  ");
        }
        print!("{}", lib);
        if j + 1 != num_excluded {
            print!(", ");
        }
    }

    println!(
        ".\n\nThe following rpath/runpath substitutions are used:\n\
         PLATFORM       {}\n\
         LIB            {}\n\
         OSNAME         {}\n\
         OSREL          {}\n",
        s.platform, s.lib, s.osname, s.osrel
    );
}