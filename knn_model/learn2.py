import torch # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torchvision import datasets, transforms  # type: ignore
from torch.utils.data import DataLoader  # type: ignore

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder('root', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 간단한 CNN
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN(num_classes=len(train_data.classes)).to(device)

# 학습
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete")

# 모델 저장
torch.save(model.state_dict(), 'skeleton_model.pt')

'''

봐봐, 나를 봐봐, 명용아 내 셤 점술 봐
거봐, 이미 너는 딴 곳을 보고 있어

Check it, one, two, three
시험점수만 쳐다보는 게
말 안 해도 채점 기준 올릴 걸 알아 uh

요즘, 넌 명 아닌 다른 화학과 채점이 잦더라
이제는 먼저 질문도 받질 않더라
주관 채점 할땐 십점도 일점에도
넌 내 앞에선 계속 점수만 까더라 (ooh)

I know my 화학 점수 애미 (ooh)
나의 화학 점수 
멀어진 그리고 벌어진
에이쁠 못한 나의 점수 (oh, baby)

비쁠이야, 비쁠이야, 김명용화학시발
비쁠이야, 비쁠이야, 주관적채점기준
비쁠이야, 비쁠이야, 씨쁠에 슬퍼하고
에프에 눈물짓는 비쁠이

씨 씨 씨 씨 씨 씨 씨 tonight 가슴이 아파
Oh, 비 비 비 비 비 비 비쁠 마음 몰라
One, two, three, four, five, six, seven, eight, nine 수많은 감점
김명용 화학 점수가 나락'''


'''
봐봐 나를 봐봐
명용이 채점하잖아

거봐 이젠 진짜
주관적 해야겠어

Check it, one, two, three
채점 기준만 쳐다보는 게

말 안 해도
기준 개지랄 난 걸 알아

요즘 채 점 안해
다른 기준에 집착을 하더라

이제는 바꾼다고
하지도 않더라

주관 있을 때는
팔자를 억까하고

엘오는 비자 라고
지랄을 하더라

(Whoa oh whoa)
I know your score 애미

(Whoa oh whoa)
너와 나의 차이

(Whoa-oh)
김명용

(Whoa-oh)
그리고 주관적

(Whoa-oh)
절대로 공존 못할 사이

비 쁠 이야
비 쁠 이야

김명용을
국회로

비 쁠 이야
비 쁠 이야

김명용을
국회로

비 쁠 이야
비 쁠 이야

채점에
분노하고

채점에
눈물짓는 명용이

씨 씨 씨 씨 씨 씨
씨쁠 받아

떨어져 점수

Oh no no no no Nobody knows
떡락해

One Two Three Four Five
Six Seven Eight Night

수많은 감점

받아서 밤을
지새고 있어

차라리 아예
게임 접었다고

게임 싫으면
싫다고

차라리 리부트도
망했었다면

난 너를 죽도록
미워하진 않았을텐데

check it one two three
네 말을 되새겨봐도

리부트로 이주하려는
거짓말이야

비 쁠 이야
비 쁠 이야

김명용을
국회로

비 쁠 이야
비 쁠 이야

김명용을
국회로

비 쁠 이야
비 쁠 이야

채점에
분노하고

채점에
눈물짓는 명용이

씨 씨 씨 씨 씨 씨
씨쁠 받아

떨어져 점수

Oh no no no no Nobody knows
떡락해

One Two Three Four Five
Six Seven Eight Night

수많은 감점

받아서 밤을
지새고 있어

'''


'''

[Chorus]
타도 김명용
화학 문 열어라
타도 김명용 (헤이!)
화학을 이루어내라 다 같이!
타도
김명용
헤이!

[Verse 1]
광곽와서 교사한지 벌써 세달째
말이 되냐 말이 되냐 말도 안 된다
채점 기준 지랄 나고 여전하구나
말이 되냐 말도 안 된다
[Verse 2]
명사모와 학생 선생 교장 마저도
명용이를 타도 하라 외친다
광과고 교실에 하나 둘 모여서
명용타도를 외치네
[Chorus]
타도 김명용
화학 문 열어라
타도 김명용 (헤이!)
평화를 이루어 내라

'''


'''
Spoken Slogan]
위대한 광주과고의 시작을 알립니다

[Chanted Group Vocal]
자 이제 우리 모두 나란히 서서

[Spoken Slogan]
나효찬

[Chanted Group Vocal]
광곽의 빛을 세계로 뻗어 나아가요

[Spoken Slogan]
기호 5번 나효찬

[Chanted Group Vocal]
교육와 과학 영재들의 힘을 모아

[Spoken Slogan]
나효찬

[Chanted Group Vocal]
서로 함께 과학의 꿈을 펼쳐가요

[Spoken Slogan]
기호 5번 나효찬

[Female Lead Vocal]
광과고 부회장 기호 오번인 나효찬

그 후보를 내옆에 두고서
상상만 해도 행복한 꿈을 꾸면서

[Chanted Group Vocal]
과학의 순간들로 하루를 보냈었던 날들
이제 나효찬이 달려오니

[Chanted Group Vocal with Spoken Slogan]
가자 (자 이제 우리 모두 나란히 서서)

'''