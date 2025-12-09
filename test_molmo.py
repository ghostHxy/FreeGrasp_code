#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时测试脚本：测试 Molmo 修复是否生效"""

from molmo_eval import process_image

print("🧪 开始测试 Molmo 处理...")
try:
    process_image(
        'data/real_examples/hard/1/image.png',
        'Point out the objects in the red rectangle on the table.',
        'data/real_examples/hard/1'
    )
    print('✅ Molmo 处理成功！')
except Exception as e:
    print(f'❌ 错误: {e}')
    import traceback
    traceback.print_exc()


