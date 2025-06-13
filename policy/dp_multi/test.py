import sys
sys.path.append('../..')
sys.path.insert(0, '../Multi-Diffusion-Policy')

print('[TEST] Importing modules...')
try:
    from script.eval_policy_multi_dp import DP
    print('[TEST] Modules imported successfully')
except Exception as e:
    print(f'[ERROR] Import failed: {e}')
    exit(1)

print('[TEST] Creating DP instance...')
try:
    dp = DP('transparent_cup_place', 'L515', 300, 300, 0)
    print('[TEST] DP instance created successfully!')
except Exception as e:
    print(f'[ERROR] DP creation failed: {e}')
    import traceback
    print(traceback.format_exc())
    exit(1)

print('[TEST] All tests passed!')