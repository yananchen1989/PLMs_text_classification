
# 384. Shuffle an Array
class Solution:

    def __init__(self, nums: List[int]):
        self.nums = nums
        self.ori = nums.copy()
    def reset(self) -> List[int]:
        return self.ori

    def shuffle(self) -> List[int]:
        for i in range(len(self.nums)):
            j = random.randrange(i, len(self.nums)) #*
            self.nums[j], self.nums[i] = self.nums[i], self.nums[j] #*
        return self.nums 

# 67. Add Binary
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        a_ = a[::-1] #*
        b_ = b[::-1]

        maxlen = max(len(a), len(b))
        if len(a) < len(b):
            a_ = a_ + '0' * (len(b)-len(a))
        elif len(b) < len(a):
            b_ = b_ + '0' * (len(a)-len(b))
        
        result = []
        c = 0
        for i in range(maxlen):
            tmp = int(a_[i]) + int(b_[i]) + c
            if tmp == 2:
                result.append('0')
                c = 1
            elif tmp == 3:
                result.append('1')
                c = 1 
            else:
                result.append(str(tmp))
                c = 0
        if c == 1:
            result.append('1')
        return ''.join(result)[::-1]



# 69. Sqrt(x)
class Solution:
    def mySqrt(self, x: int) -> int:
        l, r, ans = 0, x, -1
        while l <= r:
            mid = (l + r) // 2
            if mid * mid <= x:
                ans = mid
                l = mid + 1
            else:
                r = mid - 1
        return ans

class Solution:
    def mySqrt(self, x: int) -> int:
        if x<= 1:
            return x 
        ans = int(math.exp(0.5 * math.log(x)))
        if (ans + 1) ** 2 <= x:
            return ans + 1
        else:
            return ans 


# 70. Climbing Stairs
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# 1 <= n <= 45
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n 
        solu = [1, 2]
        for i in range(3, n+1):
            ans = solu[-1] + solu[-2]
            solu.append(ans)
        return solu[-1]





# 66. Plus One
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in range(len(digits)-1, -1, -1): $*
            digits[i] = (digits[i] + 1) % 10 
            if digits[i] != 0:
                return digits 
        return [1] + digits



# 83. Remove Duplicates from Sorted List
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head:
            return head

        cur = head
        while cur.next:
            if cur.val == cur.next.val:
                cur.next = cur.next.next
            else:
                cur = cur.next

        return head


# 21. Merge Two Sorted Lists
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        if not l1 :
            return l2 
        if not l2:
            return l1 
        
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2) 
            return l1 
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2 



# 141. check if there is a cycle
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False

# 160. Intersection of Two Linked Lists
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None 
        
        pa = headA
        seenA = set()
        while pa:
            seenA.add(pa)
            pa = pa.next 
        
        pb = headB
        while pb:
            if pb in seenA:
                return pb 
            pb = pb.next
        return None 

# 203. Remove Linked List Elements
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head 
        tmp = dummy
        
        while tmp.next:
            if tmp.next.val == val:
                tmp.next = tmp.next.next  
            else:
                tmp = tmp.next
        return dummy.next  

# 206. Reverse Linked List
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur = None 
        pre = head 
        while pre: 
            tmp = pre.next # store tmp
            pre.next = cur # 反指
            cur = pre # cur 前进
            pre = tmp  # pre 前进
        return cur  



# 234. Palindrome Linked List
# 先赋值到list，然后判断
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        p = head 
        ll = []
        while p:
            ll.append(p.val)
            p = p.next 
        return ll == ll[::-1]


class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        ans = list()
        while columnNumber > 0:
            columnNumber -= 1
            ans.append(chr(columnNumber % 26 + ord("A")))
            columnNumber //= 26
        return "".join(ans[::-1])


