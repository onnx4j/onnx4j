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

import org.junit.Test;
import org.onnx4j.tensor.DataType;
import org.onnx4j.tensor.Shape;
import org.onnx4j.tensor.TensorBuilder;

/**
 * Unit test for class of TensorManager
 */
public class TensorManagerTest {

	/**
	 * @throws Exception
	 * 
	 */
	@Test
	public void test() throws Exception {
		for (int n = 0; n < 10; n++) {
			TensorBuilder.builder(DataType.FLOAT, Shape.create(1000000L), Tensor.options()).build();
		}
	}

}