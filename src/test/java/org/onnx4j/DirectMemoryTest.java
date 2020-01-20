/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import java.nio.ByteBuffer;
import java.util.Random;

import org.junit.Test;

import sun.nio.ch.DirectBuffer;

/**
 * Unit test for Direct memory allocation and deallocation
 */
@SuppressWarnings("restriction")
public class DirectMemoryTest {

	/**
	 * <p>
	 * 说明：<br />
	 * 1、设定VM的最大堆外内存大小为100M；<br />
	 * 2、申请100M的堆外内存，并将实例引用保存到buf变量；<br />
	 * 3、调用cleaner释放内存；<br />
	 * 4、将buf变量置为空；<br />
	 * 5、申请100M的堆外内存，并将实例引用保存到buf2变量；<br />
	 * 
	 * <p>
	 * 现象观察：<br />
	 * 不会触发Full GC，表明堆外内存已方式，有足够的内存空间分配给buf2变量。
	 * 
	 * <p>
	 * 【VM参数：-XX:+PrintGCDetails -XX:MaxDirectMemorySize=100M】
	 */
	@Test
	public void testWithDeallocateByManually() {
		for (int n = 0; n < 100; n++) {
			Float magicNum = new Random().nextFloat();
			ByteBuffer buf = ByteBuffer.allocateDirect(100 * 1024 * 1024);
			buf.putFloat(0, magicNum);
			assertEquals(magicNum, Float.valueOf(buf.getFloat(0)));
			((DirectBuffer) buf).cleaner().clean();
			((DirectBuffer) buf).cleaner().clean();
			// assertFalse(magicNum.equals(Float.valueOf(buf.getFloat(0))));
		}
	}

	/**
	 * <p>
	 * 说明：<br />
	 * 1、设定VM的最大堆外内存大小为100M；<br />
	 * 2、申请100M的堆外内存，并将实例引用保存到buf变量；<br />
	 * 3、调用cleaner释放内存；<br />
	 * 4、将buf变量置为空；<br />
	 * 5、申请100M的堆外内存，并将实例引用保存到buf2变量；<br />
	 * 
	 * <p>
	 * 现象观察：<br />
	 * 虽然buf对象已被置空，但还没有触发内存回收，所以再次申请100M堆外内存时内存不足，触发系统Full GC尝试回收内存。
	 * 这时内存回收成功，能顺利再次分配100M堆外内存给buf2对象。
	 * 
	 * <p>
	 * [GC (System.gc()) [PSYoungGen: 1310K->32K(76288K)] 2319K->1040K(251392K),
	 * 0.0045025 secs] [Times: user=0.00 sys=0.00, real=0.00 secs]
	 * <p>
	 * [Full GC (System.gc()) [PSYoungGen: 32K->0K(76288K)] [ParOldGen:
	 * 1008K->1008K(175104K)] 1040K->1008K(251392K), [Metaspace:
	 * 5103K->5103K(1056768K)], 0.0050926 secs] [Times: user=0.00 sys=0.00,
	 * real=0.01 secs]
	 * 
	 * <p>
	 * 【VM参数：-XX:+PrintGCDetails -XX:MaxDirectMemorySize=100M】
	 */
	@Test
	public void testWithoutDeallocate() {
		for (int n = 0; n < 100; n++) {
			ByteBuffer buf = ByteBuffer.allocateDirect(100 * 1024 * 1024);
			buf = null;
			ByteBuffer buf2 = ByteBuffer.allocateDirect(100 * 1024 * 1024);
			assertNotNull(buf2);
		}
	}

}