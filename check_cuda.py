if __name__ == '__main__': 
    from icecream import ic
    import torch
    ic(torch.cuda.is_available())
    print(f'{torch.cuda.is_available()=}')