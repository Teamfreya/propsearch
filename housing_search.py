import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import json
from datetime import datetime
import re

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from openai import OpenAI
from swarm import Agent
from swarm.repl import run_demo_loop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class HousingCriteria:
    """Housing search criteria."""
    location: str
    max_price: int
    min_bedrooms: int
    property_type: str = "all"
    min_size_m2: Optional[int] = None
    max_size_m2: Optional[int] = None
    furnished: Optional[bool] = None
    immediate_availability: Optional[bool] = None
    pets_allowed: Optional[bool] = None

class HousingSearchAgent:
    """Housing search agent specialized for boligportal.dk."""
    
    def __init__(self):
        """Initialize the housing search agent."""
        load_dotenv()
        self._init_clients()
        self.agents = self._init_agents()
        self.base_url = "https://www.boligportal.dk/en"
        self.last_criteria = None
        self.last_url = None

    def _init_clients(self):
        """Initialize API clients."""
        try:
            api_key = self._get_env_var("FIRECRAWL_API_KEY")
            logger.info("Initializing FirecrawlApp...")
            self.firecrawl = FirecrawlApp(api_key=api_key)
            logger.info("FirecrawlApp initialized successfully")
            
            self.openai = OpenAI(api_key=self._get_env_var("OPENAI_API_KEY"))
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {str(e)}")
            raise

    @staticmethod
    def _get_env_var(name: str) -> str:
        """Safely get environment variable."""
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Missing environment variable: {name}")
        return value

    def search_housing(self, query: str) -> str:
        """Parse query and search for housing."""
        try:
            # Parse the natural language query into criteria
            criteria = self._parse_query(query)
            self.last_criteria = criteria
            
            # Perform the search
            search_results = self._execute_search(criteria)
            
            # Format and return the results
            return self._construct_response(search_results.get("listings", []))
            
        except Exception as e:
            logger.error(f"Error in search_housing: {str(e)}")
            return f"Sorry, I encountered an error. You can try searching directly at: {self.last_url}"

    def _parse_query(self, query: str) -> HousingCriteria:
        """Parse natural language query into housing criteria."""
        try:
            system_prompt = """Extract housing search criteria from the query.
            Return a JSON object with:
            - location: city name
            - max_price: maximum price in DKK (number only)
            - min_bedrooms: number of rooms (number only)
            - property_type: apartment
            
            Example: "apartment in copenhagen 2 rooms under 19000dkk per month"
            Should return: {
                "location": "copenhagen",
                "max_price": 19000,
                "min_bedrooms": 2,
                "property_type": "apartment"
            }"""

            response = self.openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            )
            
            criteria_dict = json.loads(response.choices[0].message.content)
            return HousingCriteria(**criteria_dict)
            
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            raise ValueError("Please specify location, number of rooms, and maximum monthly rent.")

    def _execute_search(self, criteria: HousingCriteria) -> Dict[str, Any]:
        """Execute the housing search with given criteria."""
        search_url = self.construct_search_url(criteria)
        self.last_url = search_url
        logger.info(f"Starting housing search: {search_url}")

        try:
            # Crawl the search results
            crawl_status = self._perform_crawl(search_url)
            
            # Process and analyze results
            listings = self._process_housing_results(crawl_status, criteria)
            
            return {
                "status": "success",
                "criteria": vars(criteria),
                "listings": listings,
                "metadata": {
                    "search_url": search_url,
                    "listings_found": len(listings),
                    "search_date": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error during housing search: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "criteria": vars(criteria),
                "search_url": search_url
            }

    def construct_search_url(self, criteria: HousingCriteria) -> str:
        """Construct the search URL based on criteria."""
        # Convert location to URL format
        location = criteria.location.lower().replace(" ", "-")
        
        # Build the base search URL
        url = f"{self.base_url}/rental-properties/{location}"
        url += f"/{criteria.min_bedrooms}-rooms"
        
        # Add parameters
        params = []
        params.append(f"max_monthly_rent={criteria.max_price}")
        if criteria.property_type != "all":
            params.append(f"housing_type={criteria.property_type}")
        if criteria.min_size_m2:
            params.append(f"min_size_m2={criteria.min_size_m2}")
        
        # Combine URL with parameters
        if params:
            url += "/?" + "&".join(params)
            
        return url

    def _perform_crawl(self, url: str) -> Dict[str, Any]:
        """Perform the web crawl."""
        try:
            crawl_response = self.firecrawl.crawl_url(
                url,
                params={
                    'limit': 15,
                    'scrapeOptions': {
                        'formats': ['text', 'links']
                    }
                },
                poll_interval=5
            )
            
            logger.info(f"Crawl completed for URL: {url}")
            return crawl_response.get('data', [])
            
        except Exception as e:
            logger.error(f"Error during crawl: {str(e)}")
            raise

    def _process_housing_results(self, crawl_data: List[Dict], 
                               criteria: HousingCriteria) -> List[Dict]:
        """Process and filter housing listings."""
        listings = []
        
        for item in crawl_data:
            try:
                # Extract basic info from text and links
                text_content = item.get('text', '')
                links = item.get('links', [])
                
                if not text_content:
                    continue
                
                # Look for price and room info in text
                price_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:DKK|kr\.)', text_content)
                room_match = re.search(r'(\d+)\s*(?:room|rm|bedroom|værelser)', text_content, re.IGNORECASE)
                size_match = re.search(r'(\d+)\s*m²', text_content)
                
                # Clean and convert values
                price = None
                if price_match:
                    price_str = price_match.group(1).replace(',', '').replace('.', '')
                    try:
                        price = float(price_str)
                    except ValueError:
                        pass
                
                rooms = int(room_match.group(1)) if room_match else None
                size = float(size_match.group(1)) if size_match else None
                
                # Find listing URL
                listing_url = next((link for link in links if '/rental-properties/' in link), None)
                
                listing = {
                    "title": text_content.split('\n')[0],
                    "price_dkk": price,
                    "location": criteria.location,
                    "size_m2": size,
                    "bedrooms": rooms,
                    "property_type": "apartment",
                    "listing_url": listing_url
                }
                
                if listing_url and not listing_url.startswith('http'):
                    listing_url = f"{self.base_url}{listing_url}"
                
                if self._matches_criteria(listing, criteria):
                    listings.append(listing)
                    
            except Exception as e:
                logger.error(f"Error processing listing: {str(e)}")
                continue
        
        return listings

    def _matches_criteria(self, listing: Dict[str, Any], criteria: HousingCriteria) -> bool:
        """Check if listing matches search criteria."""
        try:
            price = listing.get('price_dkk')
            bedrooms = listing.get('bedrooms')
            location = listing.get('location')

            if not all([price, bedrooms, location]):
                return False

            if price > criteria.max_price:
                return False
            if bedrooms < criteria.min_bedrooms:
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Error matching criteria: {str(e)}")
            return False

    def _construct_response(self, listings: List[Dict]) -> str:
        """Construct a user-friendly response from the listings."""
        if not listings:
            return (f"No available properties found matching your criteria. "
                    f"You can check for new listings at:\n{self.last_url}")
        
        response = f"Found {len(listings)} matching properties:\n\n"
        for i, listing in enumerate(listings, 1):
            response += f"{i}. {listing.get('title', 'Unlisted Property')}\n"
            if listing.get('price_dkk'):
                response += f"   Price: {listing['price_dkk']:,.0f} DKK/month\n"
            if listing.get('size_m2'):
                response += f"   Size: {listing['size_m2']} m²\n"
            if listing.get('bedrooms'):
                response += f"   Rooms: {listing['bedrooms']}\n"
            if listing.get('listing_url'):
                response += f"   Link: {listing['listing_url']}\n"
            response += "\n"
        
        return response

    def _init_agents(self) -> Dict[str, Agent]:
        """Initialize the agent system."""
        agents = {
            "ui": Agent(
                name="Housing Search Interface Agent",
                instructions="""Help users search for housing in Denmark. Start by asking for:
                - Location (e.g., Copenhagen, Aarhus)
                - Number of rooms needed
                - Maximum monthly rent in DKK
                Be friendly and helpful.""",
                functions=[self._handoff_to_search_agent]
            ),
            "searcher": Agent(
                name="Housing Search Agent",
                instructions="""Search and analyze housing listings based on user criteria.
                Present results clearly with pricing, location, and features.""",
                functions=[self.search_housing]
            )
        }
        return agents

    def _handoff_to_search_agent(self):
        """Hand off control to the search agent."""
        return self.agents["searcher"]

    def run(self):
        """Run the housing search agent."""
        try:
            run_demo_loop(self.agents["ui"], stream=True)
        except KeyboardInterrupt:
            logger.info("Shutting down housing search agent...")
        except Exception as e:
            logger.error(f"Error running housing search agent: {str(e)}")

def main():
    """Main entry point."""
    try:
        logger.info("Starting Housing Search Agent...")
        agent = HousingSearchAgent()
        logger.info("Agent initialized successfully. Starting run loop...")
        agent.run()
    except Exception as e:
        logger.error(f"Failed to start housing search agent: {str(e)}")
        raise

if __name__ == "__main__":
    main()