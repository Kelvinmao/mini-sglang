from __future__ import annotations

from minisgl.scheduler.decode import DecodeManager


def test_schedule_next_batch_returns_none_when_empty() -> None:
    manager = DecodeManager(page_size=4)

    assert not manager.runnable
    assert manager.schedule_next_batch() is None


def test_schedule_next_batch_orders_requests_by_uid(make_req) -> None:
    manager = DecodeManager(page_size=4)
    manager.filter_reqs([make_req(3), make_req(1), make_req(2)])

    batch = manager.schedule_next_batch()

    assert batch is not None
    assert batch.is_decode
    assert [req.uid for req in batch.reqs] == [1, 2, 3]


def test_filter_reqs_keeps_only_requests_that_can_decode(make_req) -> None:
    manager = DecodeManager(page_size=4)
    runnable_req = make_req(1, output_len=2)
    finished_req = make_req(2, output_len=1)
    finished_req.complete_one()

    manager.filter_reqs([runnable_req, finished_req])

    assert runnable_req in manager.running_reqs
    assert finished_req not in manager.running_reqs


def test_remove_and_abort_discard_running_requests(make_req) -> None:
    manager = DecodeManager(page_size=4)
    req1 = make_req(1)
    req2 = make_req(2)
    manager.filter_reqs([req1, req2])

    manager.remove_req(req1)
    aborted = manager.abort_req(req2.uid)

    assert req1 not in manager.running_reqs
    assert aborted is req2
    assert req2 not in manager.running_reqs
    assert manager.abort_req(99) is None


def test_inflight_tokens_reserves_remaining_decode_and_page_slack(make_req) -> None:
    manager = DecodeManager(page_size=4)
    req1 = make_req(1, output_len=5)
    req2 = make_req(2, output_len=2)
    manager.filter_reqs([req1, req2])

    expected_remaining_tokens = req1.remain_len + req2.remain_len
    expected_page_slack = (manager.page_size - 1) * 2

    assert manager.inflight_tokens == expected_remaining_tokens + expected_page_slack
