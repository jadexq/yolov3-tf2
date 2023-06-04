# pylint: disable=c-extension-no-member

import os
from pathlib import Path
from typing import Any, Dict, List

from lxml import etree


LABELS_DIR = './data/simulated/labels'
SPLITS = ['train_seg', 'val_seg']
ANNOTATIONS_DIR = './data/simulated/annotations'


def dict2xml(element, data) -> None:
    """Recursively convert dict to xml element"""
    if isinstance(data, dict):
        for key, value in data.items():
            child = etree.SubElement(element, key)
            dict2xml(child, value)
    elif isinstance(data, list):
        for item in data:
            assert isinstance(item, dict)
            for key, value in item.items():
                child = etree.SubElement(element, key)
                dict2xml(child, value)
    else:
        element.text = str(data)



def main() -> None:

    for split in SPLITS:
        labels_split_dir: str = os.path.join(LABELS_DIR, split)
        annotations_split_dir: str = os.path.join(ANNOTATIONS_DIR, split)
        Path(annotations_split_dir).mkdir(parents=True, exist_ok=True)

        for dir_path, _dir_names, file_names in os.walk(labels_split_dir):
            for file_name in file_names:

                file_path: str = os.path.join(dir_path, file_name)
                file_stem: str = file_name.split('.')[0]

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines: List[str] = f.readlines()

                objects: List[Dict[str, Any]] = []
                for line in lines:
                    line_split: List[str] = line.strip().split(' ')

                    label: str = line_split[0]
                    x_center: int = int(line_split[1])
                    y_center: int = int(line_split[2])
                    width: int = int(line_split[3])
                    height: int = int(line_split[4])

                    x_min: int = x_center - width // 2
                    y_min: int = y_center - height // 2
                    x_max: int = x_center + width // 2
                    y_max: int = y_center + height // 2

                    objects.append({
                        'name': label,
                        'bndbox': {
                            'xmin': x_min,
                            'ymin': y_min,
                            'xmax': x_max,
                            'ymax': y_max,
                        },
                    })

                xml_data: Dict[str, Any] = {
                    'filename': f'{file_stem}.npy',
                    'object': objects,
                    'size': {
                        'width': 512,
                        'height': 512,
                        'depth': 6,
                    }
                }

                xml_root = etree.Element('annotation')
                dict2xml(xml_root, xml_data)
                xml_tree = etree.ElementTree(xml_root)
                xml_tree.write(
                    os.path.join(annotations_split_dir, f'{file_stem}.xml'),
                    encoding='utf-8',
                    xml_declaration=False,
                    pretty_print=True,
                )


if __name__ == '__main__':
    main()
