from src.agent.tools.calculator import calculate

def test_calculator_basic():
    assert calculate("2+2")["result"] == 4.0
    assert calculate("10*5")["result"] == 50.0
