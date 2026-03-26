from tools import ToolRegistry


def _dummy_spec(name):
    return {
        "type": "function",
        "name": name,
        "description": f"Test tool {name}",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


class TestToolRegistry:

    def test_add_and_contains(self):
        reg = ToolRegistry()
        reg.add(_dummy_spec("foo"), lambda: "ok")
        assert "foo" in reg
        assert "bar" not in reg

    def test_getitem(self):
        fn = lambda: "result"
        reg = ToolRegistry()
        reg.add(_dummy_spec("foo"), fn)
        entry = reg["foo"]
        assert entry["fn"] is fn
        assert entry["spec"]["name"] == "foo"

    def test_iter(self):
        reg = ToolRegistry()
        reg.add(_dummy_spec("a"), lambda: None)
        reg.add(_dummy_spec("b"), lambda: None)
        assert set(reg) == {"a", "b"}

    def test_get_specs_all(self):
        reg = ToolRegistry()
        reg.add(_dummy_spec("a"), lambda: None)
        reg.add(_dummy_spec("b"), lambda: None)
        specs = reg.get_specs()
        assert len(specs) == 2
        names = {s["name"] for s in specs}
        assert names == {"a", "b"}

    def test_get_specs_filtered(self):
        reg = ToolRegistry()
        reg.add(_dummy_spec("a"), lambda: None)
        reg.add(_dummy_spec("b"), lambda: None)
        specs = reg.get_specs(["a"])
        assert len(specs) == 1
        assert specs[0]["name"] == "a"

    def test_get_specs_ignores_missing_names(self):
        reg = ToolRegistry()
        reg.add(_dummy_spec("a"), lambda: None)
        specs = reg.get_specs(["a", "nonexistent"])
        assert len(specs) == 1

    def test_subset(self):
        reg = ToolRegistry()
        reg.add(_dummy_spec("a"), lambda: None)
        reg.add(_dummy_spec("b"), lambda: None)
        reg.add(_dummy_spec("c"), lambda: None)
        sub = reg.subset(["a", "c"])
        assert "a" in sub
        assert "c" in sub
        assert "b" not in sub

    def test_subset_ignores_missing(self):
        reg = ToolRegistry()
        reg.add(_dummy_spec("a"), lambda: None)
        sub = reg.subset(["a", "nope"])
        assert len(list(sub)) == 1

    def test_empty_registry(self):
        reg = ToolRegistry()
        assert list(reg) == []
        assert reg.get_specs() == []
