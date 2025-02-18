Rectangle = {}

function Rectangle:new(o)
    o = o or {}
    setmetatable(o, {__index = self})
    return o
end

function Rectangle:area()
    return self.width * self.height
end

rect = Rectangle:new({
    width = 10,
    height = 20,
})

print(rect:area())
