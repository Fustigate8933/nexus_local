use assert_cmd::Command;

#[test]
fn prints_help() {
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.arg("--help").assert().success().stdout(predicates::str::contains("Nexus Local"));
}

#[test]
fn index_help() {
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.args(["index", "--help"]).assert().success().stdout(predicates::str::contains("Index a directory"));
}

#[test]
fn search_help() {
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.args(["search", "--help"]).assert().success().stdout(predicates::str::contains("Search for a query"));
}

#[test]
fn explain_help() {
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.args(["explain", "--help"]).assert().success().stdout(predicates::str::contains("Explain a document"));
}

#[test]
fn invalid_command_fails() {
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.arg("foo").assert().failure();
}
