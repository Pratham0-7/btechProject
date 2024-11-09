import xml.etree.ElementTree as ET

def print_edges(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    for edge in root.findall('edge'):
        edge_id = edge.attrib['id']
        print(edge_id)

print_edges("D:/PBL-2_FInal/map.net.xml")
