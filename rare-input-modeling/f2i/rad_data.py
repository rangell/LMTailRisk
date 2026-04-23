from datasets import load_dataset


def load_rexgradient_dataset(split):
    ds = load_dataset("rajpurkarlab/ReXGradient-160K")
    return ds[split]


def main():
    ds = load_rexgradient_dataset(split='train')
    import IPython; IPython.embed()
    
    
if __name__ == "__main__":
    main()