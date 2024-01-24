import scrapy

class PubMedSpider(scrapy.Spider):
    name = "pubmed"
    start_urls = ["https://pubmed.ncbi.nlm.nih.gov/?term=heart+disease"]

    def parse(self, response):
        for article in response.css("div.results-article"):
            title = article.css("a.docsum-title::text").get()
            abstract_url = article.css("a.docsum-title::attr(href)").get()
            print("Title:", title)
            print("Abstract URL:", abstract_url)
            yield response.follow(abstract_url, self.parse_abstract, meta={"title": title})

        next_page = response.css("a.next-page::attr(href)").get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)

    def parse_abstract(self, response):
        title = response.meta["title"]
        abstract = response.css("div.abstract-content::text").get()
        full_text_url = response.css("div.full-text-links a::attr(href)").get()
        print("Title:", title)
        print("Abstract:", abstract)
        print("Full Text URL:", full_text_url)
        yield {
            "title": title,
            "abstract": abstract,
            "full_text_url": full_text_url,
        }
