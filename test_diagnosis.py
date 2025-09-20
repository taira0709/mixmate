import sys
sys.path.insert(0, '.')

try:
    from app import app
    print('✓ Flask app imported successfully')
    
    with app.test_client() as client:
        rv = client.post('/inspect')
        print('✓ /inspect status:', rv.status_code)
        
        if rv.status_code == 200:
            config = rv.get_json()
            effects = config.get('effects', [])
            print('✓ Effects found:', len(effects))
            
            for i, effect in enumerate(effects):
                name = effect.get('name', 'Unknown')
                enabled = effect.get('enabled', False)
                status = 'ON' if enabled else 'OFF'
                print(f'  {i+1}. {name}: {status}')
        else:
            print('✗ /inspect failed')
            
except Exception as e:
    print('✗ Error:', str(e))
