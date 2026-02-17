import re
import os

def clean_crawled_text(input_file, output_file):
    """
    Clean crawled website text by removing repeated headers and footers.
    """
    print(f"Reading file: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    print(f"Original file size: {len(content)} characters")
    
    # Common header patterns to remove (navigation menu)
    header_patterns = [
        r"AI/MLAI/ML Development Services.*?Portfolio Ai Ml Services",
        r"Top AI Company.*?Portfolio Ai Ml Services",
        r"AI/MLAI/ML Development Services.*?Ask Galore Chatbot",
    ]
    
    # Common footer patterns to remove (contact info and links)
    footer_patterns = [
        r"NetherlandsLandfort 64.*?Portfolio",
        r"ServicesAI/ML ServicesComputer Vision.*?Portfolio",
        r"UAE: \+971 5280 50084.*?Portfolio",
    ]
    
    # Company contact information that repeats
    contact_patterns = [
        r"UAE: \+971 5280 50084.*?info@askgalore\.com",
        r"NetherlandsLandfort 64, Lelystad 8219AL.*?Dubai, UAE",
    ]
    
    # Service menu that repeats
    service_menu_pattern = r"AI/MLAI/ML Development ServicesGenerative AI Services.*?Contact Us "
    
    cleaned_content = content
    
    # Remove repeated service menus
    print("\nRemoving repeated service menus...")
    service_menu_matches = re.findall(service_menu_pattern, content, re.DOTALL)
    if service_menu_matches:
        # Keep only one instance, remove others
        menu_text = service_menu_matches[0]
        cleaned_content = cleaned_content.replace(menu_text, "\n--- NAVIGATION MENU ---\n", 1)
        cleaned_content = cleaned_content.replace(menu_text, "")
        print(f"  Removed {len(service_menu_matches) - 1} duplicate service menus")
    
    # Remove all header patterns
    print("\nRemoving header patterns...")
    for pattern in header_patterns:
        matches = re.findall(pattern, cleaned_content, re.DOTALL)
        for match in matches:
            cleaned_content = cleaned_content.replace(match, "")
        if matches:
            print(f"  Removed {len(matches)} instances of header pattern")
    
    # Remove all footer patterns
    print("\nRemoving footer patterns...")
    for pattern in footer_patterns:
        matches = re.findall(pattern, cleaned_content, re.DOTALL)
        for match in matches:
            cleaned_content = cleaned_content.replace(match, "")
        if matches:
            print(f"  Removed {len(matches)} instances of footer pattern")
    
    # Remove contact info duplicates
    print("\nRemoving duplicate contact information...")
    for pattern in contact_patterns:
        matches = re.findall(pattern, cleaned_content, re.DOTALL)
        if matches:
            # Keep first occurrence, remove others
            first_match = matches[0]
            cleaned_content = cleaned_content.replace(first_match, "\n--- CONTACT INFO ---\n" + first_match + "\n", 1)
            for match in matches[1:]:
                cleaned_content = cleaned_content.replace(match, "")
            print(f"  Removed {len(matches) - 1} duplicate contact sections")
    
    # Remove excessive whitespace and newlines
    print("\nCleaning up whitespace...")
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)  # Max 2 newlines
    cleaned_content = re.sub(r' {2,}', ' ', cleaned_content)  # Remove multiple spaces
    
    # Remove common repeated navigation items
    print("\nRemoving repeated navigation items...")
    nav_items = [
        "Explore More", "Explore →", "Get In Touch", "Connect Now",
        "ResourcesAbout usBlogsWhitepapersPortfolio",
    ]
    
    for item in nav_items:
        count = cleaned_content.count(item)
        if count > 3:  # If appears more than 3 times, likely repeated
            # Keep first 2 occurrences, remove others
            parts = cleaned_content.split(item)
            cleaned_content = item.join(parts[:3]) + ''.join(parts[3:])
            print(f"  Reduced '{item}' from {count} to 2 occurrences")
    
    # Remove lines that are just navigation/menu text (very long lines with many services)
    print("\nRemoving long navigation lines...")
    lines = cleaned_content.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip lines that are too long and contain many service names (likely navigation)
        if len(line) > 500 and ('Services' in line and 'Development' in line and 'Hire' in line):
            continue
        filtered_lines.append(line)
    
    removed_nav_lines = len(lines) - len(filtered_lines)
    if removed_nav_lines > 0:
        print(f"  Removed {removed_nav_lines} long navigation lines")
    
    cleaned_content = '\n'.join(filtered_lines)
    
    # Final cleanup
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    cleaned_content = cleaned_content.strip()
    
    # Save cleaned content
    print(f"\nSaving cleaned content to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
    
    print(f"\nCleaning complete!")
    print(f"Original size: {len(content)} characters")
    print(f"Cleaned size: {len(cleaned_content)} characters")
    print(f"Reduced by: {len(content) - len(cleaned_content)} characters ({((len(content) - len(cleaned_content)) / len(content) * 100):.1f}%)")
    
    return cleaned_content


def main():
    # File paths
    input_file = "askgalore.com_20260210_125739.txt"
    output_file = "askgalore_cleaned.txt"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    # Clean the file
    cleaned_content = clean_crawled_text(input_file, output_file)
    
    print(f"\n{'='*60}")
    print("Cleaned file saved successfully!")
    print(f"{'='*60}")
    print(f"\nYou can now use '{output_file}' for embeddings generation.")
    print("Run: python embeddings.py")


if __name__ == "__main__":
    main()
