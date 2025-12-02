"""
Web Scraper for webpage analysis
Extracts content, metadata, forms, and page state
"""

from bs4 import BeautifulSoup
import requests
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse


class WebScraper:
    """
    Scrape and analyze webpages
    """
    
    def __init__(
        self,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ):
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
    
    def scrape_url(
        self,
        url: str,
        timeout: int = 10
    ) -> Optional[BeautifulSoup]:
        """
        Fetch and parse webpage
        
        Args:
            url: URL to scrape
            timeout: Request timeout in seconds
            
        Returns:
            BeautifulSoup object or None
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            return soup
        
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def scrape_html(self, html_content: str) -> BeautifulSoup:
        """Parse HTML string"""
        return BeautifulSoup(html_content, 'lxml')
    
    def extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """
        Extract page metadata
        
        Returns:
            Dictionary with title, description, etc.
        """
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        metadata['title'] = title_tag.get_text().strip() if title_tag else ""
        
        # Meta tags
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '')
        
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords:
            metadata['keywords'] = meta_keywords.get('content', '')
        
        # Open Graph tags
        og_title = soup.find('meta', property='og:title')
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        
        og_desc = soup.find('meta', property='og:description')
        if og_desc:
            metadata['og_description'] = og_desc.get('content', '')
        
        return metadata
    
    def extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all headings"""
        headings = {}
        
        for level in range(1, 7):  # h1 to h6
            tag_name = f'h{level}'
            tags = soup.find_all(tag_name)
            headings[tag_name] = [tag.get_text().strip() for tag in tags]
        
        return headings
    
    def extract_links(
        self,
        soup: BeautifulSoup,
        base_url: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract all links
        
        Returns:
            List of dicts with 'text', 'href', 'is_external'
        """
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            text = a_tag.get_text().strip()
            
            # Convert relative URLs to absolute
            if base_url:
                href = urljoin(base_url, href)
            
            # Check if external
            is_external = False
            if base_url:
                base_domain = urlparse(base_url).netloc
                link_domain = urlparse(href).netloc
                is_external = link_domain != base_domain and link_domain != ''
            
            links.append({
                'text': text,
                'href': href,
                'is_external': is_external
            })
        
        return links
    
    def extract_forms(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract form information
        
        Returns:
            List of form dictionaries
        """
        forms = []
        
        for form in soup.find_all('form'):
            form_info = {
                'action': form.get('action', ''),
                'method': form.get('method', 'get').upper(),
                'inputs': []
            }
            
            # Extract input fields
            for input_tag in form.find_all(['input', 'textarea', 'select']):
                input_info = {
                    'type': input_tag.get('type', input_tag.name),
                    'name': input_tag.get('name', ''),
                    'id': input_tag.get('id', ''),
                    'required': input_tag.has_attr('required')
                }
                
                # Get placeholder/label
                if input_tag.has_attr('placeholder'):
                    input_info['placeholder'] = input_tag['placeholder']
                
                form_info['inputs'].append(input_info)
            
            forms.append(form_info)
        
        return forms
    
    def extract_contact_info(self, soup: BeautifulSoup) -> Dict:
        """
        Extract contact information (emails, phones)
        
        Returns:
            Dictionary with emails and phones
        """
        import re
        
        text = soup.get_text()
        
        # Email regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = list(set(re.findall(email_pattern, text)))
        
        # Phone regex (basic)
        phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        phones = list(set(re.findall(phone_pattern, text)))
        
        return {
            'emails': emails[:5],  # Limit to 5
            'phones': phones[:5]
        }
    
    def get_page_state_report(
        self,
        url: Optional[str] = None,
        soup: Optional[BeautifulSoup] = None
    ) -> Dict:
        """
        Generate comprehensive page state report
        
        Args:
            url: URL to scrape (if soup not provided)
            soup: BeautifulSoup object (if URL not provided)
            
        Returns:
            Complete page analysis
        """
        if soup is None and url:
            soup = self.scrape_url(url)
        
        if soup is None:
            return {}
        
        report = {
            'url': url or 'N/A',
            'metadata': self.extract_metadata(soup),
            'headings': self.extract_headings(soup),
            'forms': self.extract_forms(soup),
            'links': self.extract_links(soup, base_url=url),
            'contact_info': self.extract_contact_info(soup)
        }
        
        # Check for broken elements
        report['has_forms'] = len(report['forms']) > 0
        report['has_contact'] = len(report['contact_info']['emails']) > 0 or len(report['contact_info']['phones']) > 0
        report['total_links'] = len(report['links'])
        report['external_links'] = sum(1 for link in report['links'] if link['is_external'])
        
        return report
    
    def generate_text_summary(self, report: Dict) -> str:
        """Generate human-readable summary from report"""
        lines = []
        
        lines.append(f"Webpage: {report.get('url', 'N/A')}")
        
        metadata = report.get('metadata', {})
        if metadata.get('title'):
            lines.append(f"Title: {metadata['title']}")
        if metadata.get('description'):
            lines.append(f"Description: {metadata['description']}")
        
        lines.append("")
        
        # Headings
        headings = report.get('headings', {})
        h1_tags = headings.get('h1', [])
        if h1_tags:
            lines.append(f"Main Heading: {h1_tags[0]}")
        
        # Forms
        forms = report.get('forms', [])
        if forms:
            lines.append(f"\nForms: {len(forms)} found")
            for i, form in enumerate(forms[:3], 1):
                lines.append(f"  {i}. {form['method']} to {form['action'] or '(self)'}")
        
        # Links
        total_links = report.get('total_links', 0)
        external = report.get('external_links', 0)
        lines.append(f"\nLinks: {total_links} total ({external} external)")
        
        # Contact
        contact = report.get('contact_info', {})
        if contact.get('emails'):
            lines.append(f"Contact emails: {', '.join(contact['emails'][:2])}")
        
        return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    
    # Scrape and analyze
    report = scraper.get_page_state_report(url="https://example.com")
    
    if report:
        summary = scraper.generate_text_summary(report)
        print(summary)
