"""Tests for makelaar_mcp.server."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from funda.listing import (
    Address,
    Areas,
    Characteristic,
    CharacteristicSection,
    Listing,
    Media,
    MediaItem,
    PriceChange,
    PriceHistory,
    PropertyDetails,
    Price,
    Rooms,
    Urls,
)

_FIXTURES = Path(__file__).parent / "fixtures"


def make_listing(
    *,
    tiny_id="12345678",
    global_id=None,
    title="Teststraat 1",
    city="Amsterdam",
    price=400_000,
    living_area=80,
    plot_area=None,
    rooms=None,
    bedrooms=3,
    bathrooms="1 badkamer",
    year_built=2000,
    object_type="house",
    status="available",
    energy_label="A",
    url="https://www.funda.nl/detail/koop/amsterdam/huis-12345678/",
    publication_date="2024-01-15",
    garden="Achtertuin",
    description=None,
    photo_urls=None,
):
    """Build a pyfunda 3.x Listing with sensible defaults for tests."""
    chars = []
    if bathrooms is not None:
        chars.append(Characteristic(label="Aantal badkamers", value=bathrooms))
    if garden is not None:
        chars.append(Characteristic(label="Tuin", value=garden))
    sections = (
        (CharacteristicSection(title="Indeling", items=tuple(chars)),) if chars else ()
    )
    photos = tuple(MediaItem(id=str(i), url=u) for i, u in enumerate(photo_urls or []))
    return Listing(
        tiny_id=tiny_id,
        global_id=global_id,
        address=Address(title=title, city=city),
        price=Price(amount=price, formatted=f"€ {price:,}".replace(",", ".")),
        areas=Areas(living=living_area, plot=plot_area),
        rooms=Rooms(total=rooms, bedrooms=bedrooms),
        property_details=PropertyDetails(
            object_type=object_type,
            construction_year=year_built,
            energy_label=energy_label,
            status=status,
        ),
        urls=Urls(full=url),
        media=Media(photos=photos),
        characteristics=sections,
        publication_date=publication_date,
        description=description,
    )


# ---------------------------------------------------------------------------
# search_listings
# ---------------------------------------------------------------------------


def test_search_listings_returns_list():
    """search_listings returns a list of dicts with expected keys including photo_urls."""
    from makelaar_mcp.server import search_listings

    photos = [
        "https://cloud.funda.nl/valentina_media/225/504/764.jpg",
        "https://cloud.funda.nl/valentina_media/225/504/714.jpg",
    ]
    mock_listing = make_listing(photo_urls=photos)
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.search.return_value = [mock_listing]
        result = search_listings(location="amsterdam", price_max=500_000)

    assert isinstance(result, list)
    assert len(result) == 1
    row = result[0]
    for key in (
        "id",
        "title",
        "city",
        "price",
        "living_area",
        "price_per_m2",
        "bedrooms",
        "energy_label",
        "url",
        "photo_urls",
    ):
        assert key in row, f"Missing key: {key}"
    assert row["price_per_m2"] == 400_000 // 80
    assert row["photo_urls"] == photos
    assert row["first_photo_url"] == photos[0]


def test_search_listings_maps_filters_to_pyfunda():
    """search_listings maps MCP args to pyfunda 3.x filter names."""
    from makelaar_mcp.server import search_listings

    mock_listing = make_listing()
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.search.return_value = [mock_listing]
        search_listings(
            location="amsterdam",
            offering_type="buy",
            price_min=200_000,
            price_max=500_000,
            area_min=60,
            page=1,
        )
        call = mock_client.search.call_args
    assert call.args[0] == "amsterdam"
    assert call.kwargs["category"] == "buy"
    assert call.kwargs["min_price"] == 200_000
    assert call.kwargs["max_price"] == 500_000
    assert call.kwargs["min_area"] == 60
    assert call.kwargs["page"] == 1


def test_search_listings_omits_photo_keys_when_no_photos():
    """search_listings drops photo keys (not empty lists) when there are no photos."""
    from makelaar_mcp.server import search_listings

    mock_listing = make_listing()  # no photos
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.search.return_value = [mock_listing]
        result = search_listings(location="amsterdam")
    assert "photo_urls" not in result[0]
    assert "first_photo_url" not in result[0]


def test_search_listings_rent_omits_price_per_m2():
    """search_listings suppresses price_per_m2 for rentals (meaningless monthly)."""
    from makelaar_mcp.server import search_listings

    mock_listing = make_listing(price=1_800, living_area=80)
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.search.return_value = [mock_listing]
        result = search_listings(location="amsterdam", offering_type="rent")
    assert result[0]["price_per_m2"] is None
    assert mock_client.search.call_args.kwargs["category"] == "rent"


def test_search_listings_error_returns_error_type():
    """search_listings error payload carries a coarse error_type."""
    from makelaar_mcp.server import search_listings

    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.search.side_effect = ValueError("invalid sort value")
        result = search_listings(location="amsterdam")
    assert result[0]["error_type"] == "bad_input"


def test_get_listing_includes_photo_urls():
    """get_listing result includes photo_urls from the full listing detail."""
    from makelaar_mcp.server import get_listing

    photos = ["https://cloud.funda.nl/valentina_media/225/504/764.jpg"]
    mock_listing = make_listing(photo_urls=photos)
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.listing.return_value = mock_listing
        result = get_listing(listing_id=12345678)
    assert result["photo_urls"] == photos


def test_search_listings_lowercases_location():
    """search_listings lowercases location before passing to pyfunda."""
    from makelaar_mcp.server import search_listings

    mock_listing = make_listing()
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.search.return_value = [mock_listing]
        search_listings(location="Amsterdam")
        assert mock_client.search.call_args.args[0] == "amsterdam"

    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.search.return_value = [mock_listing]
        search_listings(location=["Amsterdam", "Rotterdam"])
        assert mock_client.search.call_args.args[0] == ["amsterdam", "rotterdam"]


def test_search_listings_error_handling():
    """search_listings returns error dict when the API raises a non-transport error."""
    from makelaar_mcp.server import search_listings

    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.search.side_effect = ValueError("invalid sort value")
        result = search_listings(location="amsterdam")

    assert isinstance(result, list)
    assert "error" in result[0]


# ---------------------------------------------------------------------------
# get_listing
# ---------------------------------------------------------------------------


def test_get_listing_by_id():
    """get_listing returns a dict for a numeric listing ID."""
    from makelaar_mcp.server import get_listing

    mock_listing = make_listing()
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.listing.return_value = mock_listing
        result = get_listing(listing_id=12345678)

    assert isinstance(result, dict)
    assert result["title"] == "Teststraat 1"
    assert result["price"] == 400_000
    assert result["year_built"] == 2000
    mock_client.listing.assert_called_once_with(12345678)


def test_get_listing_by_url():
    """get_listing accepts a full funda.nl URL."""
    from makelaar_mcp.server import get_listing

    url = "https://www.funda.nl/detail/koop/amsterdam/huis-12345678/"
    mock_listing = make_listing()
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.listing.return_value = mock_listing
        result = get_listing(listing_id=url)

    assert isinstance(result, dict)
    mock_client.listing.assert_called_once_with(url)


def test_get_listing_error_handling():
    """get_listing returns error dict when API raises."""
    from makelaar_mcp.server import get_listing

    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.listing.side_effect = ValueError("not found")
        result = get_listing(listing_id=99999999)

    assert "error" in result


# ---------------------------------------------------------------------------
# get_price_history
# ---------------------------------------------------------------------------


def test_get_price_history_returns_list():
    """get_price_history returns a list of price-history dicts."""
    from makelaar_mcp.server import get_price_history

    history = PriceHistory(
        status="ok",
        changes=(
            PriceChange(
                date="15 jan 2024",
                price=400_000,
                human_price="€ 400.000",
                status="asking_price",
                source="Funda",
            ),
            PriceChange(
                date="1 mrt 2023",
                price=380_000,
                human_price="€ 380.000",
                status="asking_price",
                source="Funda",
            ),
        ),
    )
    mock_listing = make_listing()
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.listing.return_value = mock_listing
        mock_client.price_history.return_value = history
        result = get_price_history(listing_id=12345678)

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["status"] == "asking_price"
    assert result[0]["price"] == 400_000
    assert result[0]["_entry_count"] == 2


def test_get_price_history_empty():
    """get_price_history reports when there is no history."""
    from makelaar_mcp.server import get_price_history

    mock_listing = make_listing()
    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.listing.return_value = mock_listing
        mock_client.price_history.return_value = PriceHistory(status="ok", changes=())
        result = get_price_history(listing_id=12345678)

    assert result[0]["entry_count"] == 0


def test_get_price_history_error_handling():
    """get_price_history returns error dict when API raises."""
    from makelaar_mcp.server import get_price_history

    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.listing.side_effect = RuntimeError("timeout")
        result = get_price_history(listing_id=12345678)

    assert isinstance(result, list)
    assert "error" in result[0]


# ---------------------------------------------------------------------------
# compare_listings
# ---------------------------------------------------------------------------


def test_compare_listings_returns_comparison():
    """compare_listings fetches multiple listings and returns comparison rows."""
    from makelaar_mcp.server import compare_listings

    listing_a = make_listing(
        tiny_id="11111111", title="Aastraat 1", price=300_000, living_area=60
    )
    listing_b = make_listing(
        tiny_id="22222222", title="Bbstraat 2", price=500_000, living_area=100
    )

    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.listing.side_effect = [listing_a, listing_b]
        result = compare_listings(listing_ids=["11111111", "22222222"])

    assert isinstance(result, list)
    assert len(result) == 2
    for row in result:
        for key in (
            "tiny_id",
            "title",
            "city",
            "price",
            "living_area",
            "price_per_m2",
            "bedrooms",
            "bathrooms",
            "year_built",
            "energy_label",
            "garden",
            "url",
        ):
            assert key in row, f"Missing comparison key: {key}"
    assert result[0]["tiny_id"] == "11111111"
    assert result[1]["price"] == 500_000


def test_compare_listings_error_handling():
    """compare_listings returns error dict when one fetch raises."""
    from makelaar_mcp.server import compare_listings

    with patch("makelaar_mcp.server._client") as mock_client:
        mock_client.listing.side_effect = RuntimeError("API down")
        result = compare_listings(listing_ids=["11111111", "22222222"])

    assert isinstance(result, list)
    assert "error" in result[0]


# ---------------------------------------------------------------------------
# calculate_dutch_mortgage
# ---------------------------------------------------------------------------


def test_dutch_mortgage_annuity_basic_keys():
    """calculate_dutch_mortgage returns all expected keys for a basic annuity mortgage."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=400_000,
        gross_annual_income=80_000,
    )

    assert isinstance(result, dict)
    for key in (
        "property_price",
        "loan_amount",
        "mortgage_type",
        "gross_monthly_payment",
        "monthly_tax_benefit",
        "net_monthly_payment",
        "total_paid",
        "total_interest",
        "nhg_eligible",
        "transfer_tax_rate",
        "transfer_tax_amount",
        "max_mortgage",
    ):
        assert key in result, f"Missing key: {key}"


def test_dutch_mortgage_annuity_payment_math():
    """Annuity payment math is correct: P=500k, r=4.5%, n=30yr (above NHG limit)."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=500_000,
        gross_annual_income=120_000,
        annual_interest_rate_pct=4.5,
        loan_term_years=30,
    )

    assert result["loan_amount"] == 500_000
    assert result["gross_monthly_payment"] == pytest.approx(2533.43, abs=5)
    assert result["total_paid"] > result["loan_amount"]
    assert result["total_interest"] == pytest.approx(
        result["total_paid"] - result["loan_amount"], rel=1e-6
    )


def test_dutch_mortgage_zero_interest():
    """At 0% interest the monthly payment equals loan / months."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=300_000,
        gross_annual_income=80_000,
        annual_interest_rate_pct=0.0,
        loan_term_years=10,
    )

    expected = 300_000 / (10 * 12)
    assert result["gross_monthly_payment"] == pytest.approx(expected, rel=1e-6)
    assert result["total_interest"] == pytest.approx(0.0, abs=1e-6)


def test_dutch_mortgage_nhg_eligible():
    """NHG is eligible when loan ≤ €435,000 and applies premium + rate discount."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=400_000,
        gross_annual_income=80_000,
    )

    assert result["nhg_eligible"] is True
    assert result["nhg_premium"] == round(400_000 * 0.006)
    assert result["effective_interest_rate_pct"] == pytest.approx(4.5 - 0.3, abs=0.01)


def test_dutch_mortgage_nhg_not_eligible():
    """NHG is not eligible when loan > €435,000."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=500_000,
        gross_annual_income=120_000,
    )

    assert result["nhg_eligible"] is False
    assert result["nhg_premium"] == 0
    assert result["effective_interest_rate_pct"] == pytest.approx(4.5, abs=0.01)


def test_dutch_mortgage_linear_payments():
    """Linear mortgage has decreasing payments: first > last."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=300_000,
        gross_annual_income=80_000,
        mortgage_type="linear",
        annual_interest_rate_pct=4.0,
        loan_term_years=30,
    )

    assert result["mortgage_type"] == "linear"
    n = 30 * 12
    principal_monthly = 300_000 / n
    monthly_rate = 0.037 / 12
    expected_first = principal_monthly + 300_000 * monthly_rate
    assert result["gross_monthly_payment"] == pytest.approx(expected_first, abs=1)
    assert "final_monthly_payment" in result
    assert result["final_monthly_payment"] < result["gross_monthly_payment"]
    assert result["total_interest"] > 0


def test_dutch_mortgage_tax_benefit():
    """Hypotheekrenteaftrek reduces net payment below gross."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=400_000,
        gross_annual_income=80_000,
        annual_interest_rate_pct=4.5,
    )

    assert result["monthly_tax_benefit"] > 0
    assert result["net_monthly_payment"] < result["gross_monthly_payment"]
    annual_interest = 400_000 * 0.042
    eigenwoningforfait = 400_000 * 0.0035
    expected_benefit = (annual_interest - eigenwoningforfait) * 0.3697 / 12
    assert result["monthly_tax_benefit"] == pytest.approx(expected_benefit, abs=1)


def test_dutch_mortgage_starter_exemption():
    """First-time buyer age 18-34, price ≤ €510k: 0% transfer tax."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=400_000,
        gross_annual_income=80_000,
        is_first_time_buyer=True,
        buyer_age=28,
    )

    assert result["transfer_tax_rate"] == 0.0
    assert result["transfer_tax_amount"] == 0


def test_dutch_mortgage_no_starter_exemption():
    """Non-first-time buyer pays 2% transfer tax."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=400_000,
        gross_annual_income=80_000,
        is_first_time_buyer=False,
    )

    assert result["transfer_tax_rate"] == 0.02
    assert result["transfer_tax_amount"] == 8_000


def test_dutch_mortgage_max_mortgage_basic():
    """Max mortgage ≈ 4.5x combined income."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=400_000,
        gross_annual_income=80_000,
    )

    assert result["max_mortgage"] == round(80_000 * 4.5)
    assert result["max_mortgage_details"]["combined_income"] == 80_000


def test_dutch_mortgage_max_mortgage_with_partner():
    """Partner income is added 100% to combined income."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price=600_000,
        gross_annual_income=80_000,
        partner_income=60_000,
    )

    assert result["max_mortgage_details"]["combined_income"] == 140_000
    assert result["max_mortgage"] == round(140_000 * 4.5)


def test_dutch_mortgage_student_debt_reduces_max():
    """Student debt reduces max mortgage."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result_no_debt = calculate_dutch_mortgage(
        price=400_000,
        gross_annual_income=80_000,
        student_debt=0,
    )
    result_with_debt = calculate_dutch_mortgage(
        price=400_000,
        gross_annual_income=80_000,
        student_debt=30_000,
    )

    assert result_with_debt["max_mortgage"] < result_no_debt["max_mortgage"]
    expected_reduction = 30_000 * 0.0045 * 12 * 30
    assert result_no_debt["max_mortgage"] - result_with_debt[
        "max_mortgage"
    ] == pytest.approx(expected_reduction, abs=1)


# ---------------------------------------------------------------------------
# calculate_total_cost
# ---------------------------------------------------------------------------


def test_total_cost_basic_keys():
    """calculate_total_cost returns all expected keys."""
    from makelaar_mcp.server import calculate_total_cost

    result = calculate_total_cost(purchase_price=400_000)

    assert isinstance(result, dict)
    for key in (
        "purchase_price",
        "costs",
        "total_additional_costs",
        "total_including_purchase",
        "cash_needed",
    ):
        assert key in result, f"Missing key: {key}"
    assert isinstance(result["costs"], list)
    assert len(result["costs"]) >= 5  # at least the mandatory costs


def test_total_cost_starter_zero_transfer_tax():
    """First-time buyer age 18-34, price ≤ €510k: 0% transfer tax."""
    from makelaar_mcp.server import calculate_total_cost

    result = calculate_total_cost(
        purchase_price=400_000,
        is_first_time_buyer=True,
        buyer_age=28,
    )

    transfer_tax = next(c for c in result["costs"] if "transfer" in c["item"].lower())
    assert transfer_tax["amount"] == 0


def test_total_cost_investor_transfer_tax():
    """Investor pays 10.4% transfer tax."""
    from makelaar_mcp.server import calculate_total_cost

    result = calculate_total_cost(
        purchase_price=400_000,
        is_investor=True,
    )

    transfer_tax = next(c for c in result["costs"] if "transfer" in c["item"].lower())
    assert transfer_tax["amount"] == round(400_000 * 0.104)  # €41,600


def test_total_cost_with_nhg():
    """NHG premium is included when use_nhg=True."""
    from makelaar_mcp.server import calculate_total_cost

    result = calculate_total_cost(
        purchase_price=400_000,
        use_nhg=True,
        mortgage_amount=400_000,
    )

    nhg = next(c for c in result["costs"] if "nhg" in c["item"].lower())
    assert nhg["amount"] == round(400_000 * 0.006)  # €2,400


def test_total_cost_without_nhg():
    """NHG premium is NOT included when use_nhg=False."""
    from makelaar_mcp.server import calculate_total_cost

    result = calculate_total_cost(purchase_price=400_000, use_nhg=False)

    nhg_items = [c for c in result["costs"] if "nhg" in c["item"].lower()]
    assert len(nhg_items) == 0


def test_total_cost_with_buyer_agent():
    """Buyer's agent cost is included when requested."""
    from makelaar_mcp.server import calculate_total_cost

    result_without = calculate_total_cost(
        purchase_price=400_000, include_buyer_agent=False
    )
    result_with = calculate_total_cost(purchase_price=400_000, include_buyer_agent=True)

    assert (
        result_with["total_additional_costs"] > result_without["total_additional_costs"]
    )
    agent = next(
        c
        for c in result_with["costs"]
        if "makelaar" in c["item"].lower() or "agent" in c["item"].lower()
    )
    assert agent["amount"] == round(400_000 * 0.015)  # 1.5%


def test_total_cost_cash_needed_equals_additional():
    """cash_needed equals total_additional_costs (bijkomende kosten from savings)."""
    from makelaar_mcp.server import calculate_total_cost

    result = calculate_total_cost(purchase_price=400_000)

    assert result["cash_needed"] == result["total_additional_costs"]
    assert (
        result["total_including_purchase"]
        == result["purchase_price"] + result["total_additional_costs"]
    )


def test_dutch_mortgage_error_handling():
    """calculate_dutch_mortgage returns error dict on unexpected input."""
    from makelaar_mcp.server import calculate_dutch_mortgage

    result = calculate_dutch_mortgage(
        price="not a number",  # type: ignore
        gross_annual_income=80_000,
    )
    assert "error" in result


def test_total_cost_error_handling():
    """calculate_total_cost returns error dict on unexpected input."""
    from makelaar_mcp.server import calculate_total_cost

    result = calculate_total_cost(purchase_price="not a number")  # type: ignore
    assert "error" in result


# ---------------------------------------------------------------------------
# error classification
# ---------------------------------------------------------------------------


def test_classify_error_types():
    """_classify_error maps exceptions to coarse, model-actionable types."""
    from makelaar_mcp import server
    from funda import ListingNotFound, SearchError

    assert server._classify_error(ListingNotFound("gone")) == "not_found"
    assert server._classify_error(ValueError("bad")) == "bad_input"
    assert server._classify_error(SearchError("upstream")) == "upstream"
    assert (
        server._classify_error(RuntimeError("curl: (92) HTTP/2 stream 1 reset"))
        == "blocked"
    )
    assert (
        server._classify_error(SearchError("Search failed (status 403)")) == "blocked"
    )


# ---------------------------------------------------------------------------
# real-payload fixtures (network-free integration of the pyfunda 3.x parsers)
# ---------------------------------------------------------------------------


def test_detail_fixture_maps_through_helper():
    """A recorded funda.io detail payload parses and maps to the detail dict."""
    from funda.parsing import parse_listing
    from makelaar_mcp.server import _detail_dict

    raw = json.loads((_FIXTURES / "detail_43117443.json").read_text())
    listing = parse_listing(raw)
    result = _detail_dict(listing)

    assert result["id"] == "43117443"
    assert result["title"] == "Reehorst 13"
    assert result["price"] == 695_000
    assert result["living_area"] == 212
    assert result["year_built"] == 1993
    assert result["energy_label"] == "B"
    assert result["price_per_m2"] == 695_000 // 212
    assert len(result["photo_urls"]) > 0
    assert result["photo_urls"][0].startswith("https://")


def test_websearch_fixture_maps_through_helper():
    """A recorded funda.nl web-search payload parses and maps to trimmed rows."""
    from funda.parsing import parse_web_search_results
    from makelaar_mcp.server import _trim_listing

    raw = json.loads((_FIXTURES / "websearch_almere_listings.json").read_text())
    listings = parse_web_search_results(raw, "buy")
    assert listings
    row = _trim_listing(listings[0])

    assert row["id"]
    assert isinstance(row["price"], int) and row["price"] > 0
    assert row["living_area"] > 0
    assert row["price_per_m2"] == row["price"] // row["living_area"]
    assert row["url"].startswith("https://www.funda.nl/")
