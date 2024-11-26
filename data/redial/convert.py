import json

# 读取entity2id JSON文件
with open('/root/autodl-tmp/LINK-main/src/data/redial/entity2id.json', 'r', encoding='utf-8') as file:
    entity_to_id = json.load(file)

# 读取movies JSON文件
with open('/root/autodl-tmp/LINK-main/src/data/redial/movies_updated_formatted.json', 'r', encoding='utf-8') as file:
    movies = json.load(file)

# 创建新的字典，将movies中的键替换为entity2id中的ID
updated_movies = {}
for key, value in movies.items():
    if key in entity_to_id:
        updated_movies[entity_to_id[key]] = value
    else:
        updated_movies[key] = value

# 将更新后的数据保存为新的JSON文件
output_json_path = '/root/autodl-tmp/LINK-main/src/data/redial/movies_with_ids.json'
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(updated_movies, f, ensure_ascii=False, indent=4)

