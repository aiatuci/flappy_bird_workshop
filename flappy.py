from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation

FPS = 60
SCREENWIDTH  = 288.0
SCREENHEIGHT = 512.0

# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100

# gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79

# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

load_saved_pool = False
save_current_pool = True
current_pool = []
fitness = []
total_models = 50

next_pipe_x = -1
next_pipe_hole_y = -1
generation = 1

# START ADD
highest_fitness = -1
best_weights = []
# END ADD


def save_pool():
    for xi in range(total_models):
        current_pool[xi].save_weights("SavedModels/model_new" + str(xi) + ".keras")
    print("Saved current pool!")

def create_model():
    # [TODO]
    # create keras model
    model = Sequential()
    model.add(Dense(3, input_shape=(3,)))
    model.add(Activation('relu'))
    model.add(Dense(7, input_shape=(3,)))
    model.add(Activation('relu'))
    model.add(Dense(1, input_shape=(3,)))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse',optimizer='adam')

    return model

def predict_action(height, dist, pipe_height, model_num):
    global current_pool
    # The height, dist and pipe_height must be between 0 to 1 (Scaled by SCREENHEIGHT)
    height = min(SCREENHEIGHT, height) / SCREENHEIGHT - 0.5
    dist = dist / 450 - 0.5 # Max pipe distance from player will be 450
    pipe_height = min(SCREENHEIGHT, pipe_height) / SCREENHEIGHT - 0.5

    # [TODO]
    # Feed in features to the neural net
    # Reshape input
    # Get prediction from model
    neural_input = np.asarray([height,dist,pipe_height])
    neural_input = np.atleast_2d(neural_input)

    output_prob = current_pool[model_num].predict(neural_input, 1)[0]

    if(output_prob[0] <= .5):
        return 1
    return 2





# [TODO]
def model_crossover(parent1, parent2):
    # obtain parent weights
    # get random gene
    # swap genes
    global current_pool

    weight1 = current_pool[parent1].get_weights()
    weight2 = current_pool[parent2].get_weights()

    new_weight1 = weight1
    new_weight2 = weight2

    gene = random.randint(0,len(new_weight1)-1)

    new_weight1[gene] = weight2[gene]
    new_weight2[gene] = weight1[gene]

    return np.asarray([new_weight1,new_weight2])



# [TODO]
def model_mutate(weights):#,generation):
    # mutate each models weights
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if( random.uniform(0,1) > .85):
                change = random.uniform(-.5,.5)
                weights[i][j] += change
    return weights



def showGameOverScreen(crashInfo):
    # Perform genetic updates here

    global current_pool
    global fitness
    global generation
    new_weights = []
    total_fitness = 0

    # START ADD
    global highest_fitness
    global best_weights
    updated = False
    # END ADD

    # Adding up fitness of all birds
    for select in range(total_models):
        total_fitness += fitness[select]
        # START ADD
        if fitness[select] >= highest_fitness:
            updated = True
            highest_fitness = fitness[select]
            best_weights = current_pool[select].get_weights()
        # END ADD

    # REMOVE HERE
    '''
    # Scaling bird's fitness by total fitness
    for select in range(total_models):
        fitness[select] /= total_fitness
        # Add previous fitness to selected bird and store
        if select > 0:
            fitness[select] += fitness[select-1]
    '''

    # ADD HERE
    # Get top two parents
    parent1 = random.randint(0,total_models-1)
    parent2 = random.randint(0,total_models-1)

    for i in range(total_models):
        if fitness[i] >= fitness[parent1]:
            parent1 = i

    for j in range(total_models):
        if j != parent1:
            if fitness[j] >= fitness[parent2]:
                parent2 = j


    for select in range(total_models // 2):
        # [TODO]
        cross_over_weights = model_crossover(parent1,parent2)
        if updated == False:
            cross_over_weights[1] = best_weights
        mutated1 = model_mutate(cross_over_weights[0])
        mutated2 = model_mutate(cross_over_weights[0])

        new_weights.append(mutated1)
        new_weights.append(mutated2)

    # Reset fitness scores for new round
    # Set new generation weights
    for select in range(len(new_weights)):
        fitness[select] = -100
        current_pool[select].set_weights(new_weights[select])
    if save_current_pool == 1:
        save_pool()

    generation += 1
    return



# Initialize all models
for i in range(total_models):
    model = create_model()
    current_pool.append(model)
    # reset fitness score
    fitness.append(-100)


if load_saved_pool:
    for i in range(total_models):
        current_pool[i].load_weights("SavedModels/model_new"+str(i)+".keras")

# for i in range(total_models):
#     print(current_pool[i].get_weights())

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((int(SCREENWIDTH), int(SCREENHEIGHT)))
    pygame.display.set_caption('Flappy Bird')


    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die']    = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit']    = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point']  = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing']   = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:

        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        movementInfo = showWelcomeAnimation()
        global fitness
        for idx in range(total_models):
            fitness[idx] = 0
        crashInfo = mainGame(movementInfo)
        showGameOverScreen(crashInfo)


def showWelcomeAnimation():
    return {
                'playery': int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2),
                'basex': 0,
                'playerIndexGen': cycle([0, 1, 2, 1]),
            }


def mainGame(movementInfo):
    global fitness
    score = playerIndex = loopIter = 0
    playerIndexGen = movementInfo['playerIndexGen']
    playersXList = []
    playersYList = []
    for idx in range(total_models):
        playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']
        playersXList.append(playerx)
        playersYList.append(playery)
    basex = movementInfo['basex']
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    global next_pipe_x
    global next_pipe_hole_y

    next_pipe_x = lowerPipes[0]['x']
    next_pipe_hole_y = (lowerPipes[0]['y'] + (upperPipes[0]['y'] + IMAGES['pipe'][0].get_height()))/2

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playersVelY    =  []   # player's velocity along Y, default same as playerFlapped
    playerMaxVelY =  10   # max vel along Y, max descend speed
    playerMinVelY =  -8   # min vel along Y, max ascend speed
    playersAccY    =  []   # players downward accleration
    playerFlapAcc =  -9   # players speed on flapping
    playersFlapped = [] # True when player flaps
    playersState = []

    for idx in range(total_models):
        playersVelY.append(-9)
        playersAccY.append(1)
        playersFlapped.append(False)
        playersState.append(True)

    alive_players = total_models

    while True:
        for idxPlayer in range(total_models):
            if playersYList[idxPlayer] < 0 and playersState[idxPlayer] == True:
                alive_players -= 1
                playersState[idxPlayer] = False
        if alive_players == 0:
            return {
                'y': 0,
                'groundCrash': True,
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': 0,
            }
        for idxPlayer in range(total_models):
            if playersState[idxPlayer]:
                fitness[idxPlayer] += 1
        next_pipe_x += pipeVelX
        for idxPlayer in range(total_models):
            if playersState[idxPlayer]:
                if predict_action(playersYList[idxPlayer], next_pipe_x, next_pipe_hole_y, idxPlayer) == 1:
                    if playersYList[idxPlayer] > -2 * IMAGES['player'][0].get_height():
                        playersVelY[idxPlayer] = playerFlapAcc
                        playersFlapped[idxPlayer] = True
                        #SOUNDS['wing'].play()
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            """if (event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP)):
                if playery > -2 * IMAGES['player'][0].get_height():
                    playerVelY = playerFlapAcc
                    playerFlapped = True
                    SOUNDS['wing'].play()
            """

        # check for crash here, returns status list
        crashTest = checkCrash({'x': playersXList, 'y': playersYList, 'index': playerIndex},
                               upperPipes, lowerPipes)

        for idx in range(total_models):
            if playersState[idx] == True and crashTest[idx] == True:
                alive_players -= 1
                playersState[idx] = False
        if alive_players == 0:
            return {
                'y': playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': score,
                'playerVelY': 0,
            }

        # check for score
        gone_through_a_pipe = False
        for idx in range(total_models):
            if playersState[idx] == True:
                pipe_idx = 0
                playerMidPos = playersXList[idx]
                for pipe in upperPipes:
                    pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width()
                    if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                        next_pipe_x = lowerPipes[pipe_idx+1]['x']
                        next_pipe_hole_y = (lowerPipes[pipe_idx+1]['y'] + (upperPipes[pipe_idx+1]['y'] + IMAGES['pipe'][pipe_idx+1].get_height())) / 2
                        gone_through_a_pipe  = True
                        # score += 1
                        fitness[idx] += 25
                        # SOUNDS['point'].play()
                    pipe_idx += 1

        if(gone_through_a_pipe):
            score += 1

        # playerIndex basex change
        if (loopIter + 1) % 3 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 100) % baseShift)

        # player's movement
        for idx in range(total_models):
            if playersState[idx] == True:
                if playersVelY[idx] < playerMaxVelY and not playersFlapped[idx]:
                    playersVelY[idx] += playersAccY[idx]
                if playersFlapped[idx]:
                    playersFlapped[idx] = False
                playerHeight = IMAGES['player'][playerIndex].get_height()
                playersYList[idx] += min(playersVelY[idx], BASEY - playersYList[idx] - playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        # print score so player overlaps the score
        showScore(score)
        for idx in range(total_models):
            if playersState[idx] == True:
                SCREEN.blit(IMAGES['player'][playerIndex], (playersXList[idx], playersYList[idx]))

        pygame.display.update()
        FPSCLOCK.tick(FPS)

def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    global generation
    scoreDigits = [int(x) for x in list(str(score))]
    generation_digits = [int(x) for x in list(str(generation))]
    totalWidth1 = 0 # total width of all numbers to be printed
    totalWidth2 = 0

    for digit in scoreDigits:
        totalWidth1 += IMAGES['numbers'][digit].get_width()

    for digit in generation_digits:
        totalWidth2 += IMAGES['numbers'][digit].get_width()

    Xoffset1 = (SCREENWIDTH - totalWidth1) / 2
    Xoffset2 = (SCREENWIDTH - totalWidth2) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset1, SCREENHEIGHT * 0.1))
        Xoffset1 += IMAGES['numbers'][digit].get_width()

    for digit in generation_digits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset2, SCREENHEIGHT * 0.2))
        Xoffset2 += IMAGES['numbers'][digit].get_width()


def checkCrash(players, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    statuses = []
    for idx in range(total_models):
        statuses.append(False)

    for idx in range(total_models):
        statuses[idx] = False
        pi = players['index']
        players['w'] = IMAGES['player'][0].get_width()
        players['h'] = IMAGES['player'][0].get_height()
        # if player crashes into ground
        if players['y'][idx] + players['h'] >= BASEY - 1:
            statuses[idx] = True
        playerRect = pygame.Rect(players['x'][idx], players['y'][idx],
                      players['w'], players['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                statuses[idx] = True
    return statuses


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
