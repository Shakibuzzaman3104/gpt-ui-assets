**What this changes and why**

**Checks**
- [ ] `for t in tests/test_*.py; do python3 $t || echo FAIL $t; done` passes (run from `skills/gpt-ui-assets`)
- [ ] New behaviour has an offline test (API mocked)
- [ ] Docs updated where a flag, command or rule changed (`tests/test_docs.py` checks flags)
- [ ] `CHANGELOG.md` updated
