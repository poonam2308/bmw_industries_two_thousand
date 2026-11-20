from parking_rules.plate_book import PlateBook


def test_no_plate_until_seen_twice():
    b = PlateBook()
    assert b.status(1) == ('none', None)
    b.saw_plate(1)
    assert b.status(1)[0] == 'none'
    b.saw_plate(1)
    assert b.status(1) == ('unread', None)


def test_one_read_in_flight_per_sign():
    b = PlateBook()
    assert b.start_read(1)
    assert not b.start_read(1)          # still pending
    assert b.start_read(2)              # other signs are independent
    b.finish_read(1, None)
    assert b.start_read(1)


def test_majority_vote_and_settling():
    b = PlateBook(settle=2)
    for text in ('8-11 h max. 3 Std.', '8-1 h max. 3 Std.', '8-11 h max. 3 Std.'):
        assert b.start_read(7)
        b.finish_read(7, text)
    status, schedule = b.status(7)
    assert status == 'read' and schedule.describe() == '8-11 h, max. 3 h'
    assert not b.start_read(7)          # settled: two identical readings


def test_gives_up_after_max_reads():
    b = PlateBook(max_reads=3)
    for _ in range(3):
        assert b.start_read(4)
        b.finish_read(4, 'unreadable %$#')
    assert not b.start_read(4)
    b.saw_plate(4), b.saw_plate(4)
    assert b.status(4) == ('unread', None)   # stays on the safe side
