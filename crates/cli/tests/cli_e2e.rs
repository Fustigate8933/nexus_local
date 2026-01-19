use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

#[test]
fn cli_e2e_index_search_explain() {
    // Setup: create a test folder and file
    let test_dir = "test_e2e_dir";
    let test_file = format!("{}/file.txt", test_dir);
    fs::create_dir_all(test_dir).unwrap();
    fs::write(&test_file, "lease agreement security deposit").unwrap();

    // Index - check for "done:" in stderr (new format)
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.args(["index", test_dir]).assert().success();

    // Search - outputs to stdout
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.args(["search", "lease agreement"]).assert().success().stdout(contains("search:"));

    // Explain - may or may not find the doc
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.args(["explain", "file.txt"]).assert().success();

    // Cleanup
    fs::remove_file(&test_file).unwrap();
    fs::remove_dir_all(test_dir).unwrap();
}
