/*
 * author: Yunqi Li
 * contact: liyunqixa@gmail.com
 */
#ifndef _GROUP_H_
#define _GROUP_H_


/*
 * externally group interchangeable values
 */
class Group
{
public:
    Group() = default;
    ~Group() = default;
    Group(const Group &other) = delete;
    Group(Group &&other) = delete;

public:
    static void readDocsAndVecs();
};

#endif // _GROUP_H_