def get_character_names(args):
    if args.is_train:
        """
        Put the name of subdirectory in retargeting/datasets/Mixamo as [[names of group A], [names of group B]]
        """
        characters = ['Abe', 'Adam', 'Amy', 'Astra', 'Crypto', 'David', 'Dummy', 
                    'Erika Archer', 'Jackie', 'James', 'Jennifer', 'Josh', 'Kate', 
                    'Knight D Pelegrini', 'Louise', 'Martha', 'Medea By M', 'Ninja', 
                    'Olivia', 'Prisoner B', 'Racer', 'Remy', 'Steve', 'Suzie', 'Y bot']
        # characters = ['Abe', 'Adam', 'Amy']

    else:
        """
        To run evaluation successfully, number of characters in both groups must be the same. Repeat is okay.
        """
        characters = ['Mousey', 'Goblin', 'Mremireh', 'Vampire']
        tmp = characters[args.eval_seq]
        characters[args.eval_seq] = characters[0]
        characters[0] = tmp

    return characters


def create_dataset(args, character_names=None):
    from data.combined_motion import TestData, MixedData

    if args.is_train:
        return MixedData(args, character_names)
    else:
        return TestData(args, character_names)


def get_test_set():
    with open('./datasets/test_set/test_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]
        return list


def get_train_list():
    with open('./datasets/Mixamo/train_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]
        return list
