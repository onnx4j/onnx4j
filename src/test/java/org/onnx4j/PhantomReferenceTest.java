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

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;

import org.junit.Test;

/**
 * Unit test for Tensor.class
 */
public class PhantomReferenceTest {

	/**
	 * @throws Exception
	 * 
	 */
	@Test
	public void test1() throws Exception {
		// 创建一个字符串对象
		String str = new String("疯狂Java讲义");
		// 创建一个引用队列
		ReferenceQueue rq = new ReferenceQueue();
		// 创建一个虚引用，让此虚引用引用到"疯狂Java讲义"字符串
		PhantomReference pr = new PhantomReference(str, rq);
		// 切断str引用和"疯狂Java讲义"字符串之间的引用
		str = null;
		// 取出虚引用所引用的对象，并不能通过虚引用获取被引用的对象，所以此处输出null
		System.out.println(pr.get()); // ①
		// 强制垃圾回收
		System.gc();
		System.runFinalization();
		// 垃圾回收之后，虚引用将被放入引用队列中
		// 取出引用队列中最先进入队列中的引用与pr进行比较
		System.out.println(rq.poll() == pr);
	}

}