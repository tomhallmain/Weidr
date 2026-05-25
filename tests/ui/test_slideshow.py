"""UI tests for slideshow timer lifecycle via MediaNavigator."""

class TestSlideshow:
    def test_toggle_slideshow_starts_classic_timer(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        assert win.slideshow_config.slideshow_running is False
        win.media_navigator.toggle_slideshow()
        assert win.slideshow_config.slideshow_running is True
        assert win.media_navigator._slideshow_timer.isActive()

    def test_stop_slideshow_timers_clears_running_state(self, window_with_dir, qtbot):
        win, _ = window_with_dir
        win.media_navigator.toggle_slideshow()
        win.media_navigator.stop_slideshow_timers()
        win.slideshow_config.end_slideshows()
        assert win.slideshow_config.slideshow_running is False
        assert win.media_navigator._slideshow_timer.isActive() is False
