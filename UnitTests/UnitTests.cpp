#include "pch.h"
#include "CppUnitTest.h"

#include "../PocketNN/pktnn_mat.cpp"
#include "../PocketNN/pktnn_tools.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace pktnn;

namespace UnitTests
{
	TEST_CLASS(UnitTests)
	{
	public:
		
		TEST_METHOD(TestClamp)
		{
			Assert::AreEqual(clampValue(100, -50, 80) , 80);
			Assert::AreEqual(clampValue(-100, -50, 80), -50);
		}
		
		TEST_METHOD(TestFloorSqrt)
		{
			Assert::AreEqual(floorSqrt(10), 3);
			Assert::AreEqual(floorSqrt(16), 4);
		}

		TEST_METHOD(TestIntRoundLog)
		{
			Assert::AreEqual(intRoundLog(2, 9), 3);
			Assert::AreEqual(intRoundLog(2, 15), 4);

			// yShift == -log(xMax - xMin) : so that log(x - xShift) + yShift == 0
			int xShift = PKT_MIN;
			int yShift = -intRoundLog(2, PKT_MAX - PKT_MIN, true);
			Assert::AreEqual(intRoundLog(2, PKT_MAX, xShift, yShift), 0);
		}
		
	};
}
