import scrapy
from scrapy import Selector
from ..items import Item

class BiqugeSpider(scrapy.Spider):
    name = "biquge"
    allowed_domains = ["3bqg.cc"]
    start_urls = ['https://www.3bqg.cc/book/10376/']

    def parse(self, response):
        sel = Selector(response)
        print(response.request.headers['User-Agent'])
        for i  in range(2):
            chapter_url =response.xpath(f'//body/div[@class="listmain"]/dl/span/dd[{i}]/a/@href').get()
            url = response.urljoin(chapter_url)
            print(url)
            if  url.endswith(('.html')):
                yield scrapy.Request(url=response.urljoin(chapter_url), callback=self.parse_page)
        

    def parse_page(self, response):
        texts = response.css('#chaptercontent *::text').getall()
        for text in texts:
            item = Item()
            item['text'] = text
            yield item