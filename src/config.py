"""Configuration module for ESG ETF performance tracking application."""

import os

ETF_tickers = [
    "CHGX",  # Chg Fin US Lrg Cp FF Fr	        Change Finance		US
    "VEGN",  # US Vegan Climate	    	        Beyond Investing	US
    "USXF",  # iShares:ESG Adv MSCI USA	        iShares			    US
    "PHO",  # Invesco Water Res		   	        Invesco			    US
    "EMXF",  # iShares:ESG Adv MSCI EM	        iShares	    		US
    # "FOOD",  # Rize Sust Future of FoodUEUSDA   Rize			    EU
    "WOOD",  # iShares:Gl Timber		        iShares			    US
    "MOO",  # VanEck:Agribusiness		    	Van Eck			    US
    "CUT",  # Invesco MSCI Gl Tmbr		    	Invesco		    	US
    "VEGI",  # iShares:MSCI Gl Agri Pro	    	iShares			    US
    # "SPAG",  # iS Agribusiness UCITS ETF	    iShares			    EU
]

api_key = os.getenv("api_avtg")

SNP500 = "IVV"  # iShares Core S&P 500 ETF

ROLLING_WINDOW = 12  # months

covid_regimes = {
    "COVID crash": ("2020-02-15", "2020-04-30"),
    "Recovery": ("2020-05-01", "2021-06-30"),
}
