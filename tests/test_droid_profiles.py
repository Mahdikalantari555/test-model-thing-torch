import shutil
import tempfile
import pytest
from src.model.droid_manager import DroidManager

@pytest.fixture
def temp_manager():
    d = tempfile.mkdtemp()
    mgr = DroidManager(base_dir=d)
    yield mgr
    shutil.rmtree(d)

def test_droid_manager_lifecycle(temp_manager):
    assert temp_manager.list_droids() == []

    d1 = temp_manager.create_droid("geospatial")
    d2 = temp_manager.create_droid("medical")

    assert "geospatial" in temp_manager.list_droids()
    assert "medical" in temp_manager.list_droids()

    # Teach distinct domain knowledge
    d1.teach("Synthetic Aperture Radar provides day-and-night all-weather radar imaging.")
    temp_manager.save_droid("geospatial")

    d2.teach("Cardiology is the study and treatment of disorders of the heart and blood vessels.")
    temp_manager.save_droid("medical")

    # Re-instantiate a fresh manager to test disk reload
    mgr2 = DroidManager(base_dir=temp_manager.base_dir)
    loaded_geo = mgr2.get_droid("geospatial")
    loaded_med = mgr2.get_droid("medical")

    # Check isolation
    geo_reply = loaded_geo.chat("What is synthetic aperture radar?")
    assert "day-and-night all-weather" in geo_reply

    med_reply = loaded_med.chat("What is cardiology?")
    assert "disorders of the heart" in med_reply

    # Medical droid should not have radar facts
    med_geo_reply = loaded_med.chat("What is synthetic aperture radar?")
    assert "day-and-night all-weather" not in med_geo_reply

def test_droid_delete(temp_manager):
    temp_manager.create_droid("temp-bot")
    assert "temp-bot" in temp_manager.list_droids()
    deleted = temp_manager.delete_droid("temp-bot")
    assert deleted is True
    assert "temp-bot" not in temp_manager.list_droids()
