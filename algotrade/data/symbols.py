"""Stock symbol lists for training data.

This module contains static lists of stock symbols organized by index/category.
Lists are manually maintained for reproducibility.

Last updated: 2026-02-01
Note: Removed delisted symbols (ANSS acquired by SNPS, SGEN acquired by PFE, SPLK acquired by CSCO)
"""

# NASDAQ 100 Index Components
# Source: Official NASDAQ-100 constituents
# Note: Using current tickers only (e.g., META not FB, GOOGL not GOOG)
NASDAQ_100 = [
    # Technology - Large Cap
    "AAPL",   # Apple Inc.
    "MSFT",   # Microsoft Corporation
    "NVDA",   # NVIDIA Corporation
    "AVGO",   # Broadcom Inc.
    "ADBE",   # Adobe Inc.
    "CSCO",   # Cisco Systems Inc.
    "CRM",    # Salesforce Inc.
    "ORCL",   # Oracle Corporation
    "ACN",    # Accenture plc
    "INTC",   # Intel Corporation
    "AMD",    # Advanced Micro Devices Inc.
    "TXN",    # Texas Instruments Inc.
    "QCOM",   # Qualcomm Inc.
    "INTU",   # Intuit Inc.
    "AMAT",   # Applied Materials Inc.
    "MU",     # Micron Technology Inc.
    "ADI",    # Analog Devices Inc.
    "LRCX",   # Lam Research Corporation
    "KLAC",   # KLA Corporation
    "SNPS",   # Synopsys Inc.
    "CDNS",   # Cadence Design Systems Inc.
    "MRVL",   # Marvell Technology Inc.
    "NXPI",   # NXP Semiconductors N.V.
    "FTNT",   # Fortinet Inc.
    "PANW",   # Palo Alto Networks Inc.
    "CRWD",   # CrowdStrike Holdings Inc.
    "TEAM",   # Atlassian Corporation
    "WDAY",   # Workday Inc.
    "ZS",     # Zscaler Inc.
    "DDOG",   # Datadog Inc.
    # "ANSS",   # ANSYS Inc. - DELISTED (acquired by Synopsys 2024)
    "ON",     # ON Semiconductor Corporation
    
    # Internet & Communication Services
    "GOOGL",  # Alphabet Inc. Class A
    "GOOG",   # Alphabet Inc. Class C
    "META",   # Meta Platforms Inc.
    "NFLX",   # Netflix Inc.
    "ABNB",   # Airbnb Inc.
    "BKNG",   # Booking Holdings Inc.
    "MELI",   # MercadoLibre Inc.
    
    # Consumer Electronics & Electric Vehicles
    "TSLA",   # Tesla Inc.
    
    # E-Commerce & Consumer Internet
    "AMZN",   # Amazon.com Inc.
    "EBAY",   # eBay Inc.
    "JD",     # JD.com Inc.
    "PDD",    # PDD Holdings Inc.
    
    # Telecommunications
    "TMUS",   # T-Mobile US Inc.
    "CMCSA",  # Comcast Corporation
    "CHTR",   # Charter Communications Inc.
    
    # Consumer Staples
    "COST",   # Costco Wholesale Corporation
    "PEP",    # PepsiCo Inc.
    "MDLZ",   # Mondelez International Inc.
    "KDP",    # Keurig Dr Pepper Inc.
    "MNST",   # Monster Beverage Corporation
    "KHC",    # The Kraft Heinz Company
    "WBA",    # Walgreens Boots Alliance Inc.
    
    # Healthcare & Biotechnology
    "AMGN",   # Amgen Inc.
    "GILD",   # Gilead Sciences Inc.
    "VRTX",   # Vertex Pharmaceuticals Inc.
    "REGN",   # Regeneron Pharmaceuticals Inc.
    "MRNA",   # Moderna Inc.
    "BIIB",   # Biogen Inc.
    "ILMN",   # Illumina Inc.
    "DXCM",   # DexCom Inc.
    "IDXX",   # IDEXX Laboratories Inc.
    # "SGEN",   # Seagen Inc. - DELISTED (acquired by Pfizer 2023)
    "AZN",    # AstraZeneca PLC
    
    # Medical Devices & Healthcare Equipment
    "ISRG",   # Intuitive Surgical Inc.
    "ZBH",    # Zimmer Biomet Holdings Inc.
    "ALGN",   # Align Technology Inc.
    
    # Financial Services
    "PYPL",   # PayPal Holdings Inc.
    "COIN",   # Coinbase Global Inc.
    
    # Industrials
    "HON",    # Honeywell International Inc.
    "ADP",    # Automatic Data Processing Inc.
    "CSX",    # CSX Corporation
    "PCAR",   # PACCAR Inc.
    "PAYX",   # Paychex Inc.
    "FAST",   # Fastenal Company
    "VRSK",   # Verisk Analytics Inc.
    "ODFL",   # Old Dominion Freight Line Inc.
    "CSGP",   # CoStar Group Inc.
    "CPRT",   # Copart Inc.
    "ROST",   # Ross Stores Inc.
    "EXC",    # Exelon Corporation
    "XEL",    # Xcel Energy Inc.
    "AEP",    # American Electric Power Company Inc.
    "EA",     # Electronic Arts Inc.
    "TTWO",   # Take-Two Interactive Software Inc.
    "CTSH",   # Cognizant Technology Solutions Corp.
    "WBD",    # Warner Bros. Discovery Inc.
    "LULU",   # Lululemon Athletica Inc.
    "DLTR",   # Dollar Tree Inc.
    "SBUX",   # Starbucks Corporation
    "MAR",    # Marriott International Inc.
    
    # Other
    "CEG",    # Constellation Energy Corporation
    "GFS",    # GlobalFoundries Inc.
    "ARM",    # Arm Holdings plc
    "DASH",   # DoorDash Inc.
    "GEHC",   # GE HealthCare Technologies Inc.
    "CCEP",   # Coca-Cola Europacific Partners PLC
    "FANG",   # Diamondback Energy Inc.
    "SMCI",   # Super Micro Computer Inc.
    "TTD",    # The Trade Desk Inc.
    "CDW",    # CDW Corporation
    # "SPLK",   # Splunk Inc. - DELISTED (acquired by Cisco 2024)
    "ENPH",   # Enphase Energy Inc.
    "LCID",   # Lucid Group Inc.
    "RIVN",   # Rivian Automotive Inc.
    "OKTA",   # Okta Inc.
    "ZM",     # Zoom Video Communications Inc.
    "SIRI",   # Sirius XM Holdings Inc.
    "MTCH",   # Match Group Inc.
]

# Subset of NASDAQ 100 - Top 10 by market cap (useful for testing)
NASDAQ_100_TOP10 = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "AVGO",
    "COST",
    "NFLX",
]

# Subset of NASDAQ 100 - Top 25 by market cap
NASDAQ_100_TOP25 = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "AVGO",
    "COST",
    "NFLX",
    "AMD",
    "ADBE",
    "PEP",
    "CSCO",
    "TMUS",
    "INTC",
    "CMCSA",
    "AMGN",
    "TXN",
    "QCOM",
    "INTU",
    "HON",
    "ISRG",
    "SBUX",
    "BKNG",
]

# Single symbol presets for quick testing
SPY = ["SPY"]
MAJOR_ETFS = ["SPY", "QQQ", "IWM", "DIA"]


def get_symbols(name: str = "nasdaq100") -> list[str]:
    """
    Get a list of symbols by name.
    
    Args:
        name: One of "nasdaq100", "nasdaq100_top10", "nasdaq100_top25", "spy", "major_etfs"
    
    Returns:
        List of ticker symbols
    """
    symbol_lists = {
        "nasdaq100": NASDAQ_100,
        "nasdaq100_top10": NASDAQ_100_TOP10,
        "nasdaq100_top25": NASDAQ_100_TOP25,
        "spy": SPY,
        "major_etfs": MAJOR_ETFS,
    }
    
    if name.lower() not in symbol_lists:
        raise ValueError(f"Unknown symbol list: {name}. Available: {list(symbol_lists.keys())}")
    
    return symbol_lists[name.lower()]
