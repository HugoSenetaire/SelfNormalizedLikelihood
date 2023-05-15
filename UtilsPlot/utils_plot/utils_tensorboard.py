from tbparse import SummaryReader
import sys
import io
import tqdm
import copy

def get_single_value(keys, dic_params, ):
    for key in keys :
        assert key not in dic_params.keys() or len(dic_params[key]) ==0, f"Key {key} already in dic_params"
    for key in keys :
        dic_params[key] = []
        dic_params[key+'_last_step'] = []
    for i, log_dir in tqdm.tqdm(enumerate(dic_params['folder_event'])):
        reader = SummaryReader(log_dir,)
        df = reader.scalars
    
        current_keys = copy.deepcopy(keys)
        try :
            list_keys = list(df['tag'].unique())
            for key in keys :
                if key not in list_keys :
                    print(f"Key {key} not in for {dic_params['folder_event'][i]}")
                    dic_params[key].append(None)
                    dic_params[key+'_last_step'].append(None)
                    current_keys.remove(key)
        except KeyError as e:
            for key in keys :
                dic_params[key].append(None)
                dic_params[key+'_last_step'].append(None)
            continue
        
        for key in current_keys :
            df_key = df[df['tag']==key][['value', 'step']]
            if len(df_key) >= 1 :
                df_drop_na = df_key.dropna()
                if len(df_drop_na) > 1:
                    print(f"More than one value for {key} in {dic_params['folder_event'][i]}, taking the last one")
                    print(df_drop_na)
                    current_df = df_drop_na[df_drop_na['step'] == df_drop_na['step'].max()]
                else :
                    current_df = df_drop_na
                dic_params[key].append(float(current_df['value'].values))
                dic_params[key+'_last_step'].append(float(current_df['step'].values))

            elif len(df_key) == 0 :
                print(f"No value for {key} in {dic_params['folder_event'][i]}")
                dic_params[key].append(None)
                dic_params[key+'_last_step'].append(None)
    return dic_params


def get_list_value(keys, dic_params, ):
    for key in keys :
        assert key not in dic_params.keys() or len(dic_params[key]) ==0, f"Key {key} already in dic_params"
    for key in keys :
        dic_params[key] = []
        # dic_params[key+'_step'] = []
    
    for i, log_dir in tqdm.tqdm(enumerate(dic_params['folder_event'])):
        reader = SummaryReader(log_dir,)
        df = reader.scalars
    
        current_keys = copy.deepcopy(keys)
        try :
            list_keys = list(df['tag'].unique())
            for key in keys :
                if key not in list_keys :
                    print(f"Key {key} not in for {dic_params['folder_event'][i]}")
                    dic_params[key].append(None)
                    # dic_params[key+'_step'].append(None)
                    current_keys.remove(key)
        except KeyError as e:
            print("KeyError")
            for key in keys :
                dic_params[key].append(None)
                # dic_params[key+'_step'].append(None)
            continue
        
        for key in current_keys :
            df_key = df[df['tag']==key][['value', 'step']]
            #
            if len(df_key) >= 1 :
                #Rename df_key
                # df_key = df_key.rename(columns={'value':key, 'step':'step'})
                dic_params[key].append(df_key)
                # dic_params[key+'_step'].append(df_key['step'].values)

            elif len(df_key) == 0 :
                print(f"No value for {key} in {dic_params['folder_event'][i]}")
                dic_params[key].append(None)
                # dic_params[key+'_step'].append(None)
    return dic_params

    # raise NotImplementedError