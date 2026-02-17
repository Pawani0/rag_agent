import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
from datetime import datetime


class WebCrawler:
    def __init__(self, base_url, max_pages=10):
        self.base_url = base_url
        self.max_pages = max_pages
        self.visited_urls = set()
        self.domain = urlparse(base_url).netloc
        
    def is_valid_url(self, url):
        """Check if URL is valid and belongs to the same domain"""
        parsed = urlparse(url)
        return bool(parsed.netloc) and parsed.netloc == self.domain
    
    def get_page_content(self, url):
        """Fetch and parse page content"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_text(self, html_content):
        """Extract text from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_links(self, html_content, base_url):
        """Extract all links from the page"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            url = urljoin(base_url, link['href'])
            if self.is_valid_url(url):
                links.append(url)
        
        return links
    
    def crawl(self):
        """Crawl the website and collect content"""
        to_visit = [self.base_url]
        all_content = []
        
        print(f"Starting crawl of {self.base_url}")
        print(f"Maximum pages to crawl: {self.max_pages}\n")
        
        while to_visit and len(self.visited_urls) < self.max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            print(f"Crawling ({len(self.visited_urls) + 1}/{self.max_pages}): {url}")
            
            html_content = self.get_page_content(url)
            if not html_content:
                continue
            
            # Mark as visited
            self.visited_urls.add(url)
            
            # Extract text content
            text_content = self.extract_text(html_content)
            all_content.append({
                'url': url,
                'content': text_content
            })
            
            # Extract and add new links
            links = self.extract_links(html_content, url)
            for link in links:
                if link not in self.visited_urls and link not in to_visit:
                    to_visit.append(link)
        
        print(f"\nCrawling complete! Visited {len(self.visited_urls)} pages.")
        return all_content
    
    def save_to_file(self, content, filename=None):
        """Save crawled content to a text file"""
        if filename is None:
            # Create filename from domain and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.domain}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for idx, page in enumerate(content, 1):
                f.write(page['content'])
                f.write("\n\n")
        
        print(f"\nContent saved to: {filename}")
        return filename


def main():
    """Main function to run the crawler"""
    # Example usage
    print("Web Crawler")
    print("-" * 50)
    
    url = input("Enter the website URL to crawl: ").strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        max_pages = int(input("Enter maximum number of pages to crawl (default 10): ").strip() or "10")
    except ValueError:
        max_pages = 10
    
    # Create crawler instance
    crawler = WebCrawler(url, max_pages=max_pages)
    
    # Crawl the website
    content = crawler.crawl()
    
    # Save to file
    if content:
        filename = crawler.save_to_file(content)
        print(f"\n✓ Successfully crawled {len(content)} pages and saved to {filename}")
    else:
        print("\n✗ No content was crawled.")


if __name__ == "__main__":
    main()
