# tests/test_generator.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from meta_agent.generator import MetaAgentGenerator
from agents.base import AgentSpec

def test_generator_load_and_generate(tmp_path):
    # create a temporary missions.yml
    mfile = tmp_path / "missions.yml"
    mfile.write_text("- id: t1\n  title: test\n  description: 'Please classify comments'\n  difficulty: easy\n")
    mg = MetaAgentGenerator()
    specs = mg.run_from_file(str(mfile))
    assert isinstance(specs, list)
    assert len(specs) >= 2
    assert all(isinstance(s, AgentSpec) for s in specs)
