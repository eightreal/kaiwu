def bump(a1, b1, a2, b2, speed_up=False):
    """
    该函数用于判断是否撞墙
        - 第一帧不会bump
        - 第二帧开始, 如果移动距离小于500则视为撞墙
    """
    if a2 == -1 and b2 == -1:
        return False
    if a1 == -1 and b1 == -1:
        return False 

    dist = ((a1-a2)**2 + (b1-b2)**2) ** (0.5)
    if speed_up:
        max_dist = 500* 1.4
    else :
        max_dist = 500
    return dist <= max_dist

def flicker_bump(a1, b1, a2, b2):
    if a2 == -1 and b2 == -1:
        return False
    if a1 == -1 and b1 == -1:
        return False 
    dist = ((a1-a2)**2 + (b1-b2)**2) ** (0.5)
    return dist <= 8000

