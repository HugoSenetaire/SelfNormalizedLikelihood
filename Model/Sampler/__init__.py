from .nuts import NutsSampler

def get_sampler(args_dict):
    if not "sampler_params" in args_dict.keys():
        args_dict["sampler_params"] = {}

    if "sampler_name" not in args_dict.keys():
        # args_dict["sampler_name"] = "nuts"
        print("No sampler specified, using ")
        return None

    
    if args_dict["sampler_name"] == "nuts":
        return NutsSampler(input_size=args_dict['input_size'],**args_dict["sampler_params"])
    else :
        raise NotImplementedError("Sampler {} not implemented".format(args_dict["sampler"]))
    
