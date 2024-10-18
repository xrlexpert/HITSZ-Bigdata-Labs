import scrapy


class QidianSpider(scrapy.Spider):
    name = "qidian"
    allowed_domains = ["qidian.com"]
    start_urls = ["https://www.qidian.com/chapter/1027368101/658867383//C2WF946J0/probe.js?v=vc1jasc"]

    def parse(self, response):
        filename = "qidian.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.body.decode('utf-8'))
