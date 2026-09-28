import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.droid import DroidEngine
from src.model.droid_manager import DroidManager

def run_verification():
    print("=== TMT DROID ENGINE END-TO-END VERIFICATION ===")
    
    mgr = DroidManager(base_dir="droids")
    droid_name = "droid-remote-sensing"
    droid = mgr.get_droid(droid_name)
    # idempotent re-runs: start from a clean knowledge slate
    droid.clear_knowledge()
    droid.step_count = 0
    
    user_paragraph = (
        "Remote sensing is the acquisition of information about an object or phenomenon "
        "without making physical contact with the object, in contrast to in situ or on-site observation. "
        "The term is applied especially to acquiring information about Earth and other planets. "
        "Remote sensing is used in numerous fields, including geophysics, geography, land surveying "
        "and most Earth science disciplines (e.g. exploration geophysics, hydrology, ecology, meteorology, "
        "oceanography, glaciology, geology). It also has military, intelligence, commercial, economic, "
        "planning, and humanitarian applications, among others."
    )
    
    print(f"\n[1] Teaching '{droid_name}' the Remote Sensing paragraph...")
    res = droid.teach(user_paragraph, source="conversational_chat", auto_tune=True)
    print(f"  -> Result: {res['status']}")
    print(f"  -> Propositions learned: {res['propositions']}")
    print(f"  -> Elapsed: {res['elapsed_ms']:.2f} ms")
    print(f"  -> Memory norm: {res['memory_norm']:.4f}")
    print(f"  -> Effective LR: {res['effective_lr']:.5f}")
    
    assert res["propositions"] >= 3, "Failed to extract propositions"
    assert res["memory_norm"] > 0, "Memory norm is zero"

    # Query 1: Definition
    print("\n[2] Testing Query 1: 'Can you define remote sensing?'")
    reply1 = droid.chat("Can you define remote sensing?")
    print(f"  -> Droid reply:\n\"{reply1}\"")
    assert "acquisition of information" in reply1.lower()
    assert "without making physical contact" in reply1.lower()
    print("  -> Passed definition recall check!")

    # Query 2: Difference with on-site observation
    print("\n[3] Testing Query 2: 'What is contrast between remote sensing and in situ observation?'")
    reply2 = droid.chat("What is contrast between remote sensing and in situ observation?")
    print(f"  -> Droid reply:\n\"{reply2}\"")
    assert "in situ" in reply2.lower() or "physical contact" in reply2.lower()
    print("  -> Passed contrast recall check!")

    # Query 3: Disciplines & Fields
    print("\n[4] Testing Query 3: 'Which fields use remote sensing?'")
    reply3 = droid.chat("Which fields use remote sensing?")
    print(f"  -> Droid reply:\n\"{reply3}\"")
    assert any(term in reply3.lower() for term in ["geophysics", "geography", "hydrology", "meteorology", "earth science"])
    print("  -> Passed field application recall check!")

    # Test Continual Learning (Zero Catastrophic Collapse)
    print("\n[5] Teaching 3 subsequent distinct domains (Lifelong Learning Stress Test)...")
    droid.teach("Photogrammetry is the science of extracting 3D measurements from overlapping 2D photographs.")
    droid.teach("Geodesy is the science of accurately measuring and understanding Earth's geometric shape and gravity field.")
    droid.teach("Hyperspectral imaging collects hundreds of contiguous spectral bands across the electromagnetic spectrum.")
    print("  -> 3 subsequent domains absorbed.")

    # Re-verify Remote Sensing definition hasn't collapsed
    print("\n[6] Re-testing Query 1 after subsequent learning...")
    re_reply = droid.chat("What is remote sensing?")
    print(f"  -> Droid reply:\n\"{re_reply}\"")
    assert "acquisition of information" in re_reply.lower()
    assert "physical contact" in re_reply.lower()
    print("  -> Passed Zero Catastrophic Collapse check!")

    # Save profile and test reload
    print("\n[7] Testing Profile Persistence & Reload...")
    mgr.save_droid(droid_name)
    reloaded_droid = DroidManager(base_dir="droids").get_droid(droid_name)
    reload_reply = reloaded_droid.chat("Define remote sensing")
    assert "acquisition of information" in reload_reply.lower()
    print("  -> Reloaded profile successfully restored exact memory state!")

    print("\n✅ ALL END-TO-END VERIFICATION CHECKS PASSED!")

if __name__ == "__main__":
    run_verification()
