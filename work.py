import requests
import csv

headers = {
    'OMINI-API-Model': 'semantic',
    'Authorization': 'Bearer sk-LI4DjYUmxpU7aBi042DaCe9c1f0448A2B0C81749F514FdBd',
}

params = {
    'query': 'semanti',
}

response = requests.get('http://s2api.ominiai.cn/generalProxy/graph/v1/paper/arXiv:2305.15021/citations?fields=title,authors&limit=500', 
                        params=params, headers=headers)

if response.status_code == 200:
    citing_data = response.json().get('data', [])
    
    with open("citations.csv", mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Title", "Authors"])
        print(citing_data)
        
        for paper in citing_data:
            citing_paper = paper.get('citingPaper', {})
            title = citing_paper.get("title", "N/A")
            authors = ", ".join([author["name"] for author in citing_paper.get("authors", [])])
            
            writer.writerow([title, authors])
    
    print(f"成功保存 {len(citing_data)} 篇论文到 citations.csv 文件中！")
else:
    print(f"Error: 无法获取引用论文数据，状态码 {response.status_code}")
