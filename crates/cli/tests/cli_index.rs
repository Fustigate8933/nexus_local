use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn cli_index_runs_and_prints_events() {
    let mut cmd = Command::cargo_bin("cli").unwrap();
    cmd.args(["index", "."]).assert().success().stdout(contains("Indexed"));
}
