#pragma once
class MyData {
public:

    int getCurrentTimestep() const;
    void setCurrentTimestep(int value);

private:
    int currentTimestep = 0;
};

inline int MyData::getCurrentTimestep() const
{
    return currentTimestep;
}

inline void MyData::setCurrentTimestep(int value)
{
    currentTimestep = value;
}
