from sklearn.metrics import recall_score
import torch

def macro_recall(output, target):

    output = torch.split(output, [168, 11, 7], dim=1)
    preds = [torch.argmax(py, dim=1).cpu().numpy() for py in output]

    target = target.cpu().numpy()

    recall_grapheme = recall_score(preds[0], target[: ,0], average='macro')
    recall_vowel = recall_score(preds[1], target[: ,1], average='macro')
    recall_consonant = recall_score(preds[2], target[: ,2], average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])

    print(f'[RECALL] Grapheme: {recall_grapheme}   Vowel: {recall_vowel}   Consonant: {recall_consonant}')
    print(f'[RECALL] Average recall: {final_score}')

    return final_score