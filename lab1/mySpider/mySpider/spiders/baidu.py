import scrapy

class BaiduSpider(scrapy.Spider):
    name = "baidu"
    
    # 初始请求的 URL（百度搜索）
    start_urls = [
        'https://baike.baidu.com/vbaike?bk_fr=home'
    ]
    
    def parse(self, response):
        # 提取搜索结果页面中的文本内容
        results = response.css('p::text').getall()
        for result in results:
            yield {
                'snippet':  result
            }
        
        # 翻页处理：获取下一页的链接
        next_pages = response.css('a::attr(href)').getall()
        for page in next_pages:
             full_url = response.urljoin(page)
             print(full_url)
             yield scrapy.Request(full_url, callback=self.parse)
